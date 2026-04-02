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


logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def srf_audio_to_redacted(event, context):
    """GCS-triggered Cloud Function entry point."""
    bucket_name = event["bucket"]
    file_name = event["name"]
    file_size = int(event.get("size", 0))

    ext = os.path.splitext(file_name)[1].lower()
    if ext not in (".wav", ".flac", ".ogg"):
        raise ValueError(
            f"Unsupported file format: {ext}. Supported formats: .wav, .flac, .ogg"
        )

    gcs_client = storage.Client()
    blob = gcs_client.bucket(bucket_name).blob(file_name)

    sample_rate, channels, duration = _get_audio_metadata(blob, ext, file_size)
    logging.info(
        "Audio metadata — sample_rate=%s  channels=%s  duration_min=%.2f",
        sample_rate, channels, duration,
    )

    if ext == ".ogg" and channels != 2:
        raise ValueError(
            f"Unsupported .ogg channel count: {channels}. Only stereo .ogg is supported."
        )

    # --- Submit long-running STT job ----------------------------------------
    gcs_uri = f"gs://{bucket_name}/{file_name}"
    operation_name = _submit_stt_job(gcs_uri, ext, sample_rate, channels)
    logging.info("STT operation started: %s", operation_name)

    # --- Poll until the STT job is done -------------------------------------
    stt_response = _poll_stt_operation(operation_name, duration)

    # --- Parse transcript + word timings ------------------------------------
    parsed = _parse_stt_response(gcs_uri, stt_response)

    # --- DLP redaction -------------------------------------------------------
    project_id = os.environ.get("GCP_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    template_id = os.environ["DLP_TEMPLATE_ID"]
    redacted = _redact_text(parsed, project_id, template_id)

    # --- Write output to GCS ------------------------------------------------
    output_bucket = os.environ["OUTPUT_BUCKET"]
    output_blob_name = f"{os.path.basename(file_name)}_{str(uuid.uuid4())[:8]}.json"
    output_blob = gcs_client.bucket(output_bucket).blob(output_blob_name)
    output_blob.upload_from_string(
        json.dumps(redacted, ensure_ascii=False), content_type="application/json"
    )
    logging.info("Result written to gs://%s/%s", output_bucket, output_blob_name)


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
    header_bytes = blob.download_as_bytes(start=0, end=end_byte)
    buf = io.BytesIO(header_bytes)

    if ext == ".wav":
        with wave.open(buf) as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            n_frames = wf.getnframes()
        duration = (n_frames / sample_rate) / 60.0
        return sample_rate, channels, duration

    if ext == ".flac":
        buf.seek(0)
        audio = mutagen.flac.FLAC(buf)
        sample_rate = audio.info.sample_rate
        channels = audio.info.channels
        duration = audio.info.length / 60.0
        return sample_rate, channels, duration

    if ext == ".ogg":
        buf.seek(0)
        try:
            audio = mutagen.oggvorbis.OggVorbis(buf)
            sample_rate = audio.info.sample_rate
        except mutagen.oggvorbis.OggVorbisHeaderError:
            buf.seek(0)
            audio = mutagen.oggopus.OggOpus(buf)
            # OGG Opus always decodes at 48 kHz; OggOpusInfo has no sample_rate field
            sample_rate = 48000
        channels = audio.info.channels
        duration = audio.info.length / 60.0
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
    logging.info("Initial STT poll sleep: %s seconds", sleep_secs)
    time.sleep(sleep_secs)

    response = get_op.execute()
    retry_count = 10
    while retry_count > 0 and not response.get("done", False):
        retry_count -= 1
        logging.info("STT not done yet. Retries left: %s. Sleeping 120 s.", retry_count)
        time.sleep(120)
        response = get_op.execute()

    if not response.get("done", False):
        raise TimeoutError(
            f"STT operation {operation_name} did not complete after polling."
        )

    return response


# ---------------------------------------------------------------------------
# STT response parsing (ported from srflongrunjobdataflow.py)
# ---------------------------------------------------------------------------

def _parse_stt_response(filename, stt_data):
    """Extract transcript text and word-level timing from the STT response."""
    result = {
        "filename": filename,
        "transcript": None,
        "words": [],
        "dlp": [],
    }

    string_transcript = ""
    for item in stt_data.get("response", {}).get("results", []):
        alternatives = item.get("alternatives", [{}])
        if alternatives and "transcript" in alternatives[0]:
            string_transcript += alternatives[0]["transcript"] + " "
    result["transcript"] = string_transcript.rstrip()

    for item in stt_data.get("response", {}).get("results", []):
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

    return result


# ---------------------------------------------------------------------------
# DLP redaction (ported from srflongrunjobdataflow.py)
# ---------------------------------------------------------------------------

def _redact_text(data, project, template_id):
    """Use Cloud DLP to detect sensitive findings in the transcript."""
    dlp = dlp_v2.DlpServiceClient()
    parent = dlp.common_project_path(project)
    inspect_template_name = f"{parent}/inspectTemplates/{template_id}"

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
                    data["dlp"].append(finding.quote)
            except AttributeError:
                pass
    else:
        logging.info("No DLP findings.")

    return data
