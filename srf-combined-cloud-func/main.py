# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Combined Cloud Function: Audio Process + STT Long-Run + DLP Redaction

Triggered by a GCS object finalize event. For each uploaded audio file it:
  1. Extracts audio metadata (sample rate, channels, duration) from GCS.
  2. Submits a long-running Speech-to-Text recognition job.
  3. Polls the STT operation until complete.
  4. Parses the transcript and word-level timing from the STT response.
  5. Runs Google Cloud DLP to detect sensitive data in the transcript.
  6. Writes the redacted JSON result to the configured output GCS bucket.

Supported audio formats: .wav, .flac, .ogg (stereo OGG Opus only)

Environment variables (required):
  OUTPUT_BUCKET      - GCS bucket name where result JSON files are written.
  DLP_TEMPLATE_ID    - ID (not full resource name) of the DLP inspect template.

Deploy as a 2nd gen Cloud Function (Cloud Run based) with a timeout of at
least 1200 seconds to accommodate the STT polling loop for long audio files.
"""

import io
import json
import logging
import os
import time
import uuid
import wave

import mutagen.flac
import mutagen.oggvorbis
import mutagen.oggopus
from google.cloud import dlp_v2
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials


# logging.basicConfig() is a no-op if the root logger already has handlers,
# which Cloud Run's runtime does before module load. Force the configuration
# so our logging.info/warning/error calls always reach Cloud Logging via stdout.
_root_logger = logging.getLogger()
if not _root_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _root_logger.addHandler(_handler)
_root_logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def srf_audio_to_redacted(event, context):
    """GCS-triggered Cloud Function entry point."""
    fn_start = time.time()
    bucket_name = event["bucket"]
    file_name = event["name"]
    file_size = int(event.get("size", 0))
    gcs_uri = f"gs://{bucket_name}/{file_name}"

    logging.info(
        "[START] srf_audio_to_redacted | file=%s | bucket=%s | size_bytes=%d",
        file_name, bucket_name, file_size,
    )

    ext = os.path.splitext(file_name)[1].lower()
    if ext not in (".wav", ".flac", ".ogg"):
        logging.error("[STAGE 1 - VALIDATE] Unsupported format: %s", ext)
        raise ValueError(
            f"Unsupported file format: {ext}. Supported formats: .wav, .flac, .ogg"
        )
    logging.info("[STAGE 1 - VALIDATE] File format accepted: %s", ext)

    # --- Stage 2: Audio metadata --------------------------------------------
    logging.info("[STAGE 2 - METADATA] Extracting audio metadata from GCS ...")
    t = time.time()
    gcs_client = storage.Client()
    blob = gcs_client.bucket(bucket_name).blob(file_name)
    sample_rate, channels, duration = _get_audio_metadata(blob, ext, file_size)

    if ext == ".ogg" and channels != 2:
        logging.error(
            "[STAGE 2 - METADATA] Unsupported .ogg channel count: %d (stereo only)", channels
        )
        raise ValueError(
            f"Unsupported .ogg channel count: {channels}. Only stereo .ogg is supported."
        )
    logging.info(
        "[STAGE 2 - METADATA] Done (%.1fs) | sample_rate=%d Hz | channels=%d | duration=%.2f min",
        time.time() - t, sample_rate, channels, duration,
    )

    # --- Stage 3: Submit STT job --------------------------------------------
    logging.info("[STAGE 3 - STT SUBMIT] Submitting long-running STT job | uri=%s", gcs_uri)
    t = time.time()
    operation_name = _submit_stt_job(gcs_uri, ext, sample_rate, channels)
    logging.info(
        "[STAGE 3 - STT SUBMIT] Done (%.1fs) | operation=%s",
        time.time() - t, operation_name,
    )

    # --- Stage 4: Poll STT --------------------------------------------------
    logging.info("[STAGE 4 - STT POLL] Waiting for STT operation to complete ...")
    t = time.time()
    stt_response = _poll_stt_operation(operation_name, duration)
    n_results = len(stt_response.get("response", {}).get("results", []))
    logging.info(
        "[STAGE 4 - STT POLL] Done (%.1fs) | result_chunks=%d",
        time.time() - t, n_results,
    )

    # --- Stage 5: Parse transcript ------------------------------------------
    logging.info("[STAGE 5 - PARSE] Parsing STT response ...")
    t = time.time()
    parsed = _parse_stt_response(gcs_uri, stt_response)
    transcript_len = len(parsed.get("transcript") or "")
    word_count = len(parsed.get("words", []))
    logging.info(
        "[STAGE 5 - PARSE] Done (%.1fs) | transcript_chars=%d | word_timings=%d",
        time.time() - t, transcript_len, word_count,
    )
    logging.info(
        "[STAGE 5 - PARSE] Transcript preview: %.200s%s",
        parsed.get("transcript") or "",
        " ..." if transcript_len > 200 else "",
    )

    # --- Stage 6: DLP redaction ---------------------------------------------
    project_id = _get_project_id()
    template_id = os.environ["DLP_TEMPLATE_ID"]
    logging.info(
        "[STAGE 6 - DLP] Running DLP inspection | project=%s | template=%s",
        project_id, template_id,
    )
    t = time.time()
    redacted = _redact_text(parsed, project_id, template_id)
    n_findings = len(redacted.get("dlp", []))
    logging.info(
        "[STAGE 6 - DLP] Done (%.1fs) | findings=%d | quotes=%s",
        time.time() - t, n_findings, redacted.get("dlp") or "none",
    )

    # --- Stage 7: Write output to GCS ---------------------------------------
    output_bucket = os.environ["OUTPUT_BUCKET"]
    output_blob_name = f"{os.path.basename(file_name)}_{str(uuid.uuid4())[:8]}.json"
    logging.info(
        "[STAGE 7 - WRITE] Writing output to gs://%s/%s ...",
        output_bucket, output_blob_name,
    )
    t = time.time()
    output_blob = gcs_client.bucket(output_bucket).blob(output_blob_name)
    output_blob.upload_from_string(
        json.dumps(redacted, ensure_ascii=False), content_type="application/json"
    )
    logging.info(
        "[STAGE 7 - WRITE] Done (%.1fs) | output=gs://%s/%s",
        time.time() - t, output_bucket, output_blob_name,
    )

    logging.info(
        "[COMPLETE] srf_audio_to_redacted finished | file=%s | total_elapsed=%.1fs",
        file_name, time.time() - fn_start,
    )


# ---------------------------------------------------------------------------
# Audio metadata extraction (replaces ffprobe from the JS function)
# ---------------------------------------------------------------------------

def _get_audio_metadata(blob, ext, file_size):
    """
    Download the first megabyte of the audio file and extract:
      - sample_rate (int, Hz)
      - channels    (int)
      - duration    (float, minutes)

    Uses Python's built-in `wave` module for WAV files and `mutagen` for
    FLAC/OGG, avoiding any ffmpeg dependency inside a Cloud Function.
    """
    max_header = 1 * 1024 * 1024  # 1 MB is enough for any audio header
    end_byte = min(max_header, file_size) - 1
    logging.info(
        "[STAGE 2 - METADATA] Downloading header bytes: 0–%d of %d total bytes",
        end_byte, file_size,
    )
    header_bytes = blob.download_as_bytes(start=0, end=end_byte)
    buf = io.BytesIO(header_bytes)

    if ext == ".wav":
        with wave.open(buf) as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            n_frames = wf.getnframes()
        duration = (n_frames / sample_rate) / 60.0
        logging.info(
            "[STAGE 2 - METADATA] WAV | sample_rate=%d | channels=%d | frames=%d | duration=%.2f min",
            sample_rate, channels, n_frames, duration,
        )
        return sample_rate, channels, duration

    if ext == ".flac":
        buf.seek(0)
        audio = mutagen.flac.FLAC(buf)
        sample_rate = audio.info.sample_rate
        channels = audio.info.channels
        duration = audio.info.length / 60.0
        logging.info(
            "[STAGE 2 - METADATA] FLAC | sample_rate=%d | channels=%d | duration=%.2f min",
            sample_rate, channels, duration,
        )
        return sample_rate, channels, duration

    if ext == ".ogg":
        buf.seek(0)
        try:
            audio = mutagen.oggvorbis.OggVorbis(buf)
            sample_rate = audio.info.sample_rate
            logging.info("[STAGE 2 - METADATA] OGG codec detected: Vorbis")
        except mutagen.oggvorbis.OggVorbisHeaderError:
            buf.seek(0)
            audio = mutagen.oggopus.OggOpus(buf)
            # OGG Opus always decodes at 48 kHz; OggOpusInfo has no sample_rate field
            sample_rate = 48000
            logging.info("[STAGE 2 - METADATA] OGG codec detected: Opus (sample_rate fixed at 48000 Hz)")
        channels = audio.info.channels
        duration = audio.info.length / 60.0
        logging.info(
            "[STAGE 2 - METADATA] OGG | sample_rate=%d | channels=%d | duration=%.2f min",
            sample_rate, channels, duration,
        )
        return sample_rate, channels, duration

    raise ValueError(f"Unhandled extension in _get_audio_metadata: {ext}")


# ---------------------------------------------------------------------------
# Speech-to-Text long-running job submission
# ---------------------------------------------------------------------------

def _submit_stt_job(gcs_uri, ext, sample_rate, channels):
    """Submit a longRunningRecognize job and return the operation name."""
    audio = speech.RecognitionAudio(uri=gcs_uri)

    config_kwargs = dict(
        sample_rate_hertz=int(sample_rate),
        language_code="en-US",
        max_alternatives=0,
        enable_word_time_offsets=True,
        use_enhanced=True,
        audio_channel_count=int(channels),
        enable_separate_recognition_per_channel=True,
        model="phone_call",
    )
    if ext == ".ogg":
        config_kwargs["encoding"] = speech.RecognitionConfig.AudioEncoding.OGG_OPUS

    logging.info(
        "[STAGE 3 - STT SUBMIT] STT config: %s",
        {k: str(v) for k, v in config_kwargs.items()},
    )

    config = speech.RecognitionConfig(**config_kwargs)
    stt_client = speech.SpeechClient()
    operation = stt_client.long_running_recognize(config=config, audio=audio)

    # operation.operation is the underlying google.longrunning.Operation proto
    return operation.operation.name


# ---------------------------------------------------------------------------
# STT operation polling (ported from srflongrunjobdataflow.py)
# ---------------------------------------------------------------------------

def _poll_stt_operation(operation_name, duration_minutes):
    """
    Poll the STT long-running operation until it reports done.

    Initial sleep is half the audio duration (matching the Dataflow behaviour).
    After that, retry up to 10 times with 2-minute gaps.
    """
    credentials = GoogleCredentials.get_application_default()
    speech_service = discovery.build("speech", "v1p1beta1", credentials=credentials)
    get_op = speech_service.operations().get(name=operation_name)

    sleep_secs = round(float(duration_minutes) / 2 * 60) if duration_minutes else 5
    logging.info(
        "[STAGE 4 - STT POLL] Initial sleep: %d s (half of %.2f min audio)",
        sleep_secs, duration_minutes,
    )
    time.sleep(sleep_secs)

    poll_start = time.time()
    response = get_op.execute()
    retry_count = 10
    attempt = 0
    while retry_count > 0 and not response.get("done", False):
        retry_count -= 1
        attempt += 1
        elapsed = time.time() - poll_start
        logging.info(
            "[STAGE 4 - STT POLL] Attempt %d: not done yet | elapsed=%.0fs | retries_left=%d | sleeping 120s",
            attempt, elapsed, retry_count,
        )
        time.sleep(120)
        response = get_op.execute()

    if not response.get("done", False):
        logging.error(
            "[STAGE 4 - STT POLL] Timed out after %d attempts | operation=%s",
            attempt, operation_name,
        )
        raise TimeoutError(
            f"STT operation {operation_name} did not complete after polling."
        )

    logging.info(
        "[STAGE 4 - STT POLL] Operation complete | attempts=%d | total_poll_elapsed=%.0fs",
        attempt, time.time() - poll_start,
    )
    return response


# ---------------------------------------------------------------------------
# STT response parsing (ported from srflongrunjobdataflow.py)
# ---------------------------------------------------------------------------

def _parse_stt_response(filename, stt_data):
    """Extract transcript text and word-level timing from the STT response."""
    results = stt_data.get("response", {}).get("results", [])
    logging.info("[STAGE 5 - PARSE] Processing %d result chunk(s) from STT response", len(results))

    result = {
        "filename": filename,
        "transcript": None,
        "words": [],
        "dlp": [],
    }

    string_transcript = ""
    for item in results:
        alternatives = item.get("alternatives", [{}])
        if alternatives and "transcript" in alternatives[0]:
            string_transcript += alternatives[0]["transcript"] + " "
    result["transcript"] = string_transcript.rstrip()

    for item in results:
        alternatives = item.get("alternatives", [{}])
        if alternatives:
            for word in alternatives[0].get("words", []):
                result["words"].append(
                    {
                        "word": word["word"],
                        "startsecs": word["startTime"].rstrip("s"),
                        "endsecs": word["endTime"].rstrip("s"),
                    }
                )

    logging.info(
        "[STAGE 5 - PARSE] transcript_chars=%d | word_timings=%d",
        len(result["transcript"] or ""), len(result["words"]),
    )
    if result["words"]:
        first = result["words"][0]
        last = result["words"][-1]
        logging.info(
            "[STAGE 5 - PARSE] First word: '%s' @ %ss | Last word: '%s' @ %ss",
            first["word"], first["startsecs"], last["word"], last["endsecs"],
        )
    return result


# ---------------------------------------------------------------------------
# DLP redaction (ported from srflongrunjobdataflow.py)
# ---------------------------------------------------------------------------

def _redact_text(data, project, template_id):
    """Use Cloud DLP to detect sensitive findings in the transcript."""
    dlp = dlp_v2.DlpServiceClient()
    parent = dlp.common_project_path(project)
    inspect_template_name = f"{parent}/inspectTemplates/{template_id}"

    logging.info(
        "[STAGE 6 - DLP] Inspecting transcript (%d chars) | template=%s",
        len(data.get("transcript") or ""), inspect_template_name,
    )

    request = dlp_v2.InspectContentRequest(
        parent=parent,
        inspect_template_name=inspect_template_name,
        item={"value": data["transcript"]},
    )
    response = dlp.inspect_content(request=request)

    if response.result.findings:
        for finding in response.result.findings:
            try:
                if finding.quote:
                    logging.info(
                        "[STAGE 6 - DLP] Finding: info_type=%s | likelihood=%s | quote='%s'",
                        finding.info_type.name, finding.likelihood.name, finding.quote,
                    )
                    data["dlp"].append(finding.quote)
            except AttributeError:
                pass
        logging.info("[STAGE 6 - DLP] Total findings: %d", len(data["dlp"]))
    else:
        logging.info("[STAGE 6 - DLP] No findings.")

    return data
