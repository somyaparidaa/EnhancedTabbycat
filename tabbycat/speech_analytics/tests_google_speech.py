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
    """Unit tests for GoogleSpeechClient using mocked Google Cloud Speech-to-Text API."""

    def setUp(self):
        self.mock_speech_client = MagicMock(spec=speech_v1.SpeechClient)
        self.client = GoogleSpeechClient(client=self.mock_speech_client)
        # Valid 16kHz PCM WAV in memory
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(struct.pack("<h", 0) * 1600)
        self.valid_wav_bytes = buf.getvalue()
        self.dummy_audio = io.BytesIO(self.valid_wav_bytes)

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
        info = detect_audio_format(self.valid_wav_bytes, "debate.wav")
        self.assertEqual(info["format"], "wav")
        self.assertEqual(info["sample_rate"], 16000)
        self.assertEqual(info["channels"], 1)
        self.assertEqual(info["sample_width"], 2)

    def test_detect_audio_format_m4a_magic_bytes(self):
        m4a_header = b"\x00\x00\x00\x1cftypM4A \x00\x00\x00\x00M4A mp42isom"
        info = detect_audio_format(m4a_header, "speech.unknown")
        self.assertEqual(info["format"], "m4a")

    def test_detect_audio_format_fallback_by_extension(self):
        info = detect_audio_format(b"unknown_bytes_12345", "audio.m4a")
        self.assertEqual(info["format"], "m4a")

    def test_synchronous_recognition_success(self):
        # Build mock RecognizeResponse
        mock_response = speech_v1.RecognizeResponse()

        # Result 1: Speaker 1
        result1 = speech_v1.SpeechRecognitionResult()
        alt1 = speech_v1.SpeechRecognitionAlternative()
        alt1.transcript = "Mr Speaker we propose"
        alt1.confidence = 0.95

        w1 = speech_v1.WordInfo(
            word="Mr",
            start_time=datetime.timedelta(seconds=0, milliseconds=0),
            end_time=datetime.timedelta(seconds=0, milliseconds=400),
            speaker_tag=1,
            confidence=0.96,
        )
        w2 = speech_v1.WordInfo(
            word="Speaker",
            start_time=datetime.timedelta(seconds=0, milliseconds=400),
            end_time=datetime.timedelta(seconds=0, milliseconds=900),
            speaker_tag=1,
            confidence=0.94,
        )
        w3 = speech_v1.WordInfo(
            word="we",
            start_time=datetime.timedelta(seconds=0, milliseconds=900),
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

        # Result 2: Speaker 2 (POI)
        result2 = speech_v1.SpeechRecognitionResult()
        alt2 = speech_v1.SpeechRecognitionAlternative()
        alt2.transcript = "Point of information"
        alt2.confidence = 0.90

        w5 = speech_v1.WordInfo(
            word="Point",
            start_time=datetime.timedelta(seconds=2, milliseconds=0),
            end_time=datetime.timedelta(seconds=2, milliseconds=300),
            speaker_tag=2,
            confidence=0.92,
        )
        w6 = speech_v1.WordInfo(
            word="of",
            start_time=datetime.timedelta(seconds=2, milliseconds=300),
            end_time=datetime.timedelta(seconds=2, milliseconds=600),
            speaker_tag=2,
            confidence=0.88,
        )
        w7 = speech_v1.WordInfo(
            word="information",
            start_time=datetime.timedelta(seconds=2, milliseconds=600),
            end_time=datetime.timedelta(seconds=3, milliseconds=100),
            speaker_tag=2,
            confidence=0.91,
        )
        alt2.words.extend([w5, w6, w7])
        result2.alternatives.append(alt2)

        mock_response.results.extend([result1, result2])
        self.mock_speech_client.recognize.return_value = mock_response

        # Call create_batch_job
        job = self.client.create_batch_job(self.dummy_audio, language_code="en-US")

        self.assertEqual(job["status"], "COMPLETED")
        self.assertTrue(job["job_id"].startswith("google_sync_"))
        self.assertIn("transcript_payload", job)

        # Inspect recognition config passed to Google
        call_kwargs = self.mock_speech_client.recognize.call_args.kwargs
        config = call_kwargs["config"]
        self.assertEqual(config.language_code, "en-US")
        self.assertEqual(config.model, "latest_long")
        self.assertEqual(config.encoding, speech_v1.RecognitionConfig.AudioEncoding.LINEAR16)
        self.assertEqual(config.sample_rate_hertz, 16000)
        self.assertTrue(config.enable_word_time_offsets)
        self.assertTrue(config.diarization_config.enable_speaker_diarization)

        # Verify normalized transcript payload
        payload = job["transcript_payload"]
        self.assertEqual(payload["text"], "Mr Speaker we propose Point of information")
        self.assertEqual(payload["language_code"], "en-US")
        self.assertAlmostEqual(payload["overall_confidence"], 0.925, places=2)
        self.assertEqual(payload["duration"], 3.1)

        # Verify segments grouping
        segments = payload["segments"]
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["speaker"], "Speaker 1")
        self.assertEqual(segments[0]["text"], "Mr Speaker we propose")
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 1.8)

        self.assertEqual(segments[1]["speaker"], "Speaker 2")
        self.assertEqual(segments[1]["text"], "Point of information")
        self.assertEqual(segments[1]["start"], 2.0)
        self.assertEqual(segments[1]["end"], 3.1)

        # Verify fetch_completed_transcript retrieves this payload
        fetched = self.client.fetch_completed_transcript(job["job_id"])
        self.assertEqual(fetched["text"], payload["text"])

        # Verify get_job_status reports COMPLETED
        status = self.client.get_job_status(job["job_id"])
        self.assertEqual(status["status"], "COMPLETED")
        self.assertEqual(status["progress"], 100)

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

        # Mock convert_audio_to_wav to return valid WAV bytes
        with patch("speech_analytics.services.google_speech.convert_audio_to_wav") as mock_convert:
            mock_convert.return_value = (self.valid_wav_bytes, "speech_converted.wav")
            audio_input = io.BytesIO(m4a_dummy)
            audio_input.name = "recording.m4a"

            job = self.client.create_batch_job(audio_input, language_code="en-US")
            self.assertEqual(job["status"], "COMPLETED")
            mock_convert.assert_called_once()
            # Verify Recognize called with converted WAV audio config
            call_kwargs = self.mock_speech_client.recognize.call_args.kwargs
            config = call_kwargs["config"]
            self.assertEqual(config.encoding, speech_v1.RecognitionConfig.AudioEncoding.LINEAR16)
            self.assertEqual(config.sample_rate_hertz, 16000)

    def test_conversion_failure_raises_stt_configuration_error(self):
        m4a_dummy = b"\x00\x00\x00\x1cftypM4A \x00\x00\x00\x00M4A mp42isom"
        with patch("speech_analytics.services.google_speech.convert_audio_to_wav") as mock_convert:
            mock_convert.side_effect = STTConfigurationError("Conversion failed: invalid audio data")
            audio_input = io.BytesIO(m4a_dummy)
            audio_input.name = "broken.m4a"
            with self.assertRaises(STTConfigurationError) as ctx:
                self.client.create_batch_job(audio_input)
            self.assertIn("Conversion failed", str(ctx.exception))

    def test_convert_audio_to_wav_missing_tools_raises_error(self):
        with patch("shutil.which", return_value=None):
            with self.assertRaises(STTConfigurationError) as ctx:
                convert_audio_to_wav(b"dummy", "audio.m4a")
            self.assertIn("neither 'ffmpeg' nor 'afconvert' is installed", str(ctx.exception))

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

    def test_model_selection_override_setting(self):
        mock_response = speech_v1.RecognizeResponse()
        self.mock_speech_client.recognize.return_value = mock_response

        with override_settings(GOOGLE_SPEECH_MODEL="latest_short"):
            self.client.create_batch_job(self.dummy_audio)
            call_kwargs = self.mock_speech_client.recognize.call_args.kwargs
            self.assertEqual(call_kwargs["config"].model, "latest_short")

    def test_explicit_model_kwarg_takes_precedence(self):
        mock_response = speech_v1.RecognizeResponse()
        self.mock_speech_client.recognize.return_value = mock_response

        with override_settings(GOOGLE_SPEECH_MODEL="latest_long"):
            self.client.create_batch_job(self.dummy_audio, model="video")
            call_kwargs = self.mock_speech_client.recognize.call_args.kwargs
            self.assertEqual(call_kwargs["config"].model, "video")

    def test_empty_model_does_not_pass_invalid_model_field(self):
        mock_response = speech_v1.RecognizeResponse()
        self.mock_speech_client.recognize.return_value = mock_response

        with override_settings(GOOGLE_SPEECH_MODEL=""):
            self.client.create_batch_job(self.dummy_audio, model="")
            call_kwargs = self.mock_speech_client.recognize.call_args.kwargs
            # RecognitionConfig without model set should have empty string / default unset proto field
            self.assertEqual(call_kwargs["config"].model, "")

