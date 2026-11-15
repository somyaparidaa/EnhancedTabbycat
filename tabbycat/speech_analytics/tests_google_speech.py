import datetime
import io
import struct
import wave
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings
from google.api_core import exceptions as google_exceptions
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import speech_v1

from speech_analytics.services.base import (
    BaseSpeechToTextClient,
    STTAuthenticationError,
    STTConfigurationError,
    STTConnectionError,
    STTRateLimitError,
    STTTimeoutError,
)
from speech_analytics.services.factory import get_stt_client
from speech_analytics.services.google_speech import (
    GoogleSpeechClient,
    convert_audio_to_wav,
    detect_audio_format,
)


class GoogleSpeechClientTests(SimpleTestCase):
    """Unit tests for GoogleSpeechClient using mocked Google Cloud Speech-to-Text and Storage APIs."""

    def setUp(self):
        self.mock_speech_client = MagicMock(spec=speech_v1.SpeechClient)
        self.mock_storage_service = MagicMock()
        self.client = GoogleSpeechClient(
            client=self.mock_speech_client,
            storage_service=self.mock_storage_service,
        )

        # Build short (1-second) valid WAV in memory
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(struct.pack("<h", 0) * 16000)
        self.valid_short_wav = buf.getvalue()
        self.dummy_audio = io.BytesIO(self.valid_short_wav)

    def _generate_wav_bytes(self, duration_seconds: int = 1) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(struct.pack("<h", 0) * (16000 * duration_seconds))
        return buf.getvalue()

    def test_implements_base_client_interface(self):
        self.assertIsInstance(self.client, BaseSpeechToTextClient)

    def test_factory_returns_google_client(self):
        with override_settings(SPEECH_TRANSCRIPTION_PROVIDER="google"):
            with patch("speech_analytics.services.google_speech.speech_v1.SpeechClient"):
                client = get_stt_client()
                self.assertIsInstance(client, GoogleSpeechClient)

    def test_authentication_error_on_missing_adc(self):
        with patch("speech_analytics.services.google_speech.speech_v1.SpeechClient") as mock_cls:
            mock_cls.side_effect = DefaultCredentialsError("ADC not found")
            with self.assertRaises(STTAuthenticationError) as ctx:
                GoogleSpeechClient()
            self.assertIn("ADC", str(ctx.exception))

    def test_detect_audio_format_wav(self):
        info = detect_audio_format(self.valid_short_wav, "debate.wav")
        self.assertEqual(info["format"], "wav")
        self.assertEqual(info["sample_rate"], 16000)
        self.assertEqual(info["channels"], 1)
        self.assertEqual(info["sample_width"], 2)
        self.assertEqual(info["duration"], 1.0)

    def test_detect_audio_format_m4a_magic_bytes(self):
        m4a_header = b"\x00\x00\x00\x1cftypM4A \x00\x00\x00\x00M4A mp42isom"
        info = detect_audio_format(m4a_header, "speech.unknown")
        self.assertEqual(info["format"], "m4a")

    def test_detect_audio_format_fallback_by_extension(self):
        info = detect_audio_format(b"unknown_bytes_12345", "audio.m4a")
        self.assertEqual(info["format"], "m4a")

    # Short audio (<60s) synchronous recognition
    def test_synchronous_recognition_success(self):
        mock_response = speech_v1.RecognizeResponse()
        result1 = speech_v1.SpeechRecognitionResult()
        alt1 = speech_v1.SpeechRecognitionAlternative()
        alt1.transcript = "Mr Speaker we propose"
        alt1.confidence = 0.95

        w1 = speech_v1.WordInfo(
            word="Mr",
            start_time=datetime.timedelta(seconds=0),
            end_time=datetime.timedelta(milliseconds=400),
            speaker_tag=1,
            confidence=0.96,
        )
        w2 = speech_v1.WordInfo(
            word="Speaker",
            start_time=datetime.timedelta(milliseconds=400),
            end_time=datetime.timedelta(milliseconds=900),
            speaker_tag=1,
            confidence=0.94,
        )
        w3 = speech_v1.WordInfo(
            word="we",
            start_time=datetime.timedelta(milliseconds=900),
            end_time=datetime.timedelta(seconds=1, milliseconds=100),
            speaker_tag=1,
            confidence=0.95,
        )
        w4 = speech_v1.WordInfo(
            word="propose",
            start_time=datetime.timedelta(seconds=1, milliseconds=100),
            end_time=datetime.timedelta(seconds=1, milliseconds=800),
            speaker_tag=1,
            confidence=0.95,
        )
        alt1.words.extend([w1, w2, w3, w4])
        result1.alternatives.append(alt1)
        mock_response.results.append(result1)
        self.mock_speech_client.recognize.return_value = mock_response

        job = self.client.create_batch_job(self.dummy_audio, language_code="en-US")
        self.assertEqual(job["status"], "COMPLETED")
        self.assertTrue(job["job_id"].startswith("google_sync_"))
        self.assertIn("transcript_payload", job)

        call_kwargs = self.mock_speech_client.recognize.call_args.kwargs
        config = call_kwargs["config"]
        self.assertEqual(config.language_code, "en-US")
        self.assertEqual(config.model, "latest_long")
        self.assertEqual(config.encoding, speech_v1.RecognitionConfig.AudioEncoding.LINEAR16)

        # Storage upload should NOT have been invoked for short audio
        self.mock_storage_service.upload_audio_bytes.assert_not_called()

    # Long audio (7-minute speech = 420s) asynchronous recognition via GCS
    def test_long_audio_7_minute_speech_triggers_gcs_and_long_running(self):
        long_wav_bytes = self._generate_wav_bytes(duration_seconds=420)  # 7 minutes
        self.mock_storage_service.upload_audio_bytes.return_value = "gs://mock-bucket/speech_recordings/uuid/audio.wav"

        mock_operation = MagicMock()
        mock_operation.operation.name = "projects/mock/locations/global/operations/op_7min_speech"
        self.mock_speech_client.long_running_recognize.return_value = mock_operation

        audio_input = io.BytesIO(long_wav_bytes)
        audio_input.name = "substantive_speech.wav"

        job = self.client.create_batch_job(audio_input, language_code="en-US")

        # Job must be in PROCESSING state without blocking
        self.assertEqual(job["status"], "PROCESSING")
        self.assertEqual(job["job_id"], "projects/mock/locations/global/operations/op_7min_speech")
        self.assertEqual(job["gcs_uri"], "gs://mock-bucket/speech_recordings/uuid/audio.wav")

        # Verify GCS upload was called
        self.mock_storage_service.upload_audio_bytes.assert_called_once()
        # Verify long_running_recognize was called with gcs uri
        call_kwargs = self.mock_speech_client.long_running_recognize.call_args.kwargs
        self.assertEqual(call_kwargs["audio"].uri, "gs://mock-bucket/speech_recordings/uuid/audio.wav")

    # Long debate recording (60-minute debate = 3600s)
    def test_long_audio_60_minute_debate_recording(self):
        long_wav_bytes = self._generate_wav_bytes(duration_seconds=3600)  # 60 minutes
        self.mock_storage_service.upload_audio_bytes.return_value = "gs://mock-bucket/speech_recordings/uuid/full_debate.wav"

        mock_operation = MagicMock()
        mock_operation.operation.name = "projects/mock/locations/global/operations/op_debate_60m"
        self.mock_speech_client.long_running_recognize.return_value = mock_operation

        audio_input = io.BytesIO(long_wav_bytes)
        audio_input.name = "debate_round_1.wav"

        job = self.client.create_batch_job(audio_input, language_code="en-US")
        self.assertEqual(job["status"], "PROCESSING")
        self.assertEqual(job["job_id"], "projects/mock/locations/global/operations/op_debate_60m")

    # Polling and transcript retrieval for long-running operation
    def test_polling_and_fetching_completed_long_running_transcript(self):
        op_id = "projects/mock/locations/global/operations/op_test_123"
        gcs_uri = "gs://mock-bucket/speech_recordings/uuid/audio.wav"

        mock_op = MagicMock()
        mock_op.done = True
        mock_op.error = None

        mock_response = speech_v1.LongRunningRecognizeResponse()
        res = speech_v1.SpeechRecognitionResult()
        alt = speech_v1.SpeechRecognitionAlternative()
        alt.transcript = "This is a completed long debate speech transcript."
        alt.confidence = 0.94
        w1 = speech_v1.WordInfo(
            word="This",
            start_time=datetime.timedelta(seconds=0),
            end_time=datetime.timedelta(seconds=1),
            speaker_tag=1,
            confidence=0.95,
        )
        alt.words.append(w1)
        res.alternatives.append(alt)
        mock_response.results.append(res)
        mock_op.response = mock_response

        # Register operation in local store
        self.client._local_operations[op_id] = {
            'job_id': op_id,
            'status': 'PROCESSING',
            'gcs_uri': gcs_uri,
            'operation': mock_op,
            'language_code': 'en-US',
        }

        # Check status
        status = self.client.get_job_status(op_id)
        self.assertEqual(status["status"], "COMPLETED")
        self.assertEqual(status["progress"], 100)

        # Fetch completed transcript
        with override_settings(GOOGLE_CLOUD_STORAGE_CLEANUP_ON_SUCCESS=True):
            payload = self.client.fetch_completed_transcript(op_id)
            self.assertEqual(payload["text"], "This is a completed long debate speech transcript.")
            self.assertEqual(payload["overall_confidence"], 0.94)
            # Verify temporary GCS object was deleted after successful fetch
            self.mock_storage_service.delete_audio_by_uri.assert_called_with(gcs_uri)

    def test_failed_long_running_operation_reports_failure(self):
        op_id = "projects/mock/locations/global/operations/op_failed_123"
        mock_op = MagicMock()
        mock_op.done = True
        mock_op.error.message = "Audio file contains corrupt frames"

        self.client._local_operations[op_id] = {
            'job_id': op_id,
            'status': 'PROCESSING',
            'operation': mock_op,
        }

        status = self.client.get_job_status(op_id)
        self.assertEqual(status["status"], "FAILED")
        self.assertEqual(status["error"], "Audio file contains corrupt frames")

    def test_gcs_upload_failure_raises_stt_connection_error(self):
        long_wav_bytes = self._generate_wav_bytes(duration_seconds=100)
        self.mock_storage_service.upload_audio_bytes.side_effect = STTConnectionError("GCS network timeout")

        audio_input = io.BytesIO(long_wav_bytes)
        with self.assertRaises(STTConnectionError):
            self.client.create_batch_job(audio_input)

    def test_m4a_automatic_transcoding_and_recognition(self):
        m4a_dummy = b"\x00\x00\x00\x1cftypM4A \x00\x00\x00\x00M4A mp42isom" + b"\x00" * 100
        mock_response = speech_v1.RecognizeResponse()
        res = speech_v1.SpeechRecognitionResult()
        alt = speech_v1.SpeechRecognitionAlternative()
        alt.transcript = "Transcoded M4A speech"
        alt.confidence = 0.94
        res.alternatives.append(alt)
        mock_response.results.append(res)
        self.mock_speech_client.recognize.return_value = mock_response

        with patch("speech_analytics.services.google_speech.convert_audio_to_wav") as mock_convert:
            mock_convert.return_value = (self.valid_short_wav, "speech_converted.wav")
            audio_input = io.BytesIO(m4a_dummy)
            audio_input.name = "recording.m4a"

            job = self.client.create_batch_job(audio_input, language_code="en-US")
            self.assertEqual(job["status"], "COMPLETED")
            mock_convert.assert_called_once()

    def test_empty_results_normalization(self):
        mock_response = speech_v1.RecognizeResponse()
        self.mock_speech_client.recognize.return_value = mock_response

        job = self.client.create_batch_job(self.dummy_audio)
        self.assertEqual(job["status"], "COMPLETED")
        payload = job["transcript_payload"]
        self.assertEqual(payload["text"], "")
        self.assertIsNone(payload["overall_confidence"])
        self.assertIsNone(payload["duration"])
        self.assertEqual(payload["segments"], [])

    def test_rate_limit_exception_mapping(self):
        self.mock_speech_client.recognize.side_effect = google_exceptions.ResourceExhausted("Quota exceeded")
        with self.assertRaises(STTRateLimitError) as ctx:
            self.client.create_batch_job(self.dummy_audio)
        self.assertEqual(ctx.exception.status_code, 429)

    def test_timeout_exception_mapping(self):
        self.mock_speech_client.recognize.side_effect = google_exceptions.DeadlineExceeded("Deadline exceeded")
        with self.assertRaises(STTTimeoutError):
            self.client.create_batch_job(self.dummy_audio)

    def test_connection_error_mapping(self):
        self.mock_speech_client.recognize.side_effect = google_exceptions.ServiceUnavailable("Service unavailable")
        with self.assertRaises(STTConnectionError):
            self.client.create_batch_job(self.dummy_audio)

    def test_permission_denied_mapping(self):
        self.mock_speech_client.recognize.side_effect = google_exceptions.PermissionDenied("API disabled")
        with self.assertRaises(STTAuthenticationError):
            self.client.create_batch_job(self.dummy_audio)
