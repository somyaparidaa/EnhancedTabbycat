import io
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

from speech_analytics.services.base import (
    BaseSpeechToTextClient,
    STTAPIResponseError,
    STTAuthenticationError,
    STTConfigurationError,
    STTConnectionError,
    STTError,
    STTMalformedResponseError,
    STTRateLimitError,
    STTTimeoutError,
)
from speech_analytics.services.exceptions import (
    GnaniAPIResponseError,
    GnaniAuthenticationError,
    GnaniConfigurationError,
    GnaniConnectionError,
    GnaniMalformedResponseError,
    GnaniRateLimitError,
    GnaniSTTError,
    GnaniTimeoutError,
)
from speech_analytics.services.factory import get_stt_client
from speech_analytics.services.gnani import GnaniPrismaBatchClient


class DummySTTClient(BaseSpeechToTextClient):
    """Concrete implementation for testing BaseSpeechToTextClient interface contract."""

    def create_batch_job(self, audio_file, language_code="en-IN", **kwargs):
        return {"job_id": "dummy_123", "status": "CREATED"}

    def start_batch_job(self, job_id):
        return {"job_id": job_id, "status": "STARTING"}

    def get_job_status(self, job_id):
        return {"job_id": job_id, "status": "COMPLETED"}

    def fetch_completed_transcript(self, job_id):
        return {
            "text": "Hello world",
            "language_code": "en",
            "metadata": {},
            "segments": [],
        }


class SpeechToTextBaseInterfaceTests(SimpleTestCase):
    """Verify the BaseSpeechToTextClient ABC and exception hierarchy."""

    def test_cannot_instantiate_base_client_directly(self):
        with self.assertRaises(TypeError):
            BaseSpeechToTextClient()

    def test_concrete_subclass_implements_interface(self):
        client = DummySTTClient()
        self.assertIsInstance(client, BaseSpeechToTextClient)
        job = client.create_batch_job(io.BytesIO(b"dummy audio"))
        self.assertEqual(job["job_id"], "dummy_123")
        self.assertEqual(job["status"], "CREATED")

        started = client.start_batch_job("dummy_123")
        self.assertEqual(started["status"], "STARTING")

        status = client.get_job_status("dummy_123")
        self.assertEqual(status["status"], "COMPLETED")

        transcript = client.fetch_completed_transcript("dummy_123")
        self.assertEqual(transcript["text"], "Hello world")

    def test_gnani_exceptions_inherit_from_provider_neutral_stt_exceptions(self):
        self.assertTrue(issubclass(GnaniSTTError, STTError))
        self.assertTrue(issubclass(GnaniConfigurationError, STTConfigurationError))
        self.assertTrue(issubclass(GnaniAuthenticationError, STTAuthenticationError))
        self.assertTrue(issubclass(GnaniRateLimitError, STTRateLimitError))
        self.assertTrue(issubclass(GnaniAPIResponseError, STTAPIResponseError))
        self.assertTrue(issubclass(GnaniConnectionError, STTConnectionError))
        self.assertTrue(issubclass(GnaniTimeoutError, STTTimeoutError))
        self.assertTrue(issubclass(GnaniMalformedResponseError, STTMalformedResponseError))

    def test_rate_limit_error_supports_retry_after(self):
        err = STTRateLimitError("Rate limit exceeded", status_code=429, retry_after=30)
        self.assertEqual(err.status_code, 429)
        self.assertEqual(err.retry_after, 30)

        gnani_err = GnaniRateLimitError("Gnani rate limited", retry_after=60)
        self.assertIsInstance(gnani_err, STTRateLimitError)
        self.assertEqual(gnani_err.retry_after, 60)

    def test_gnani_client_is_instance_of_base_speech_to_text_client(self):
        self.assertTrue(issubclass(GnaniPrismaBatchClient, BaseSpeechToTextClient))


class SpeechToTextFactoryTests(SimpleTestCase):
    """Verify provider selection and fallback via get_stt_client()."""

    @override_settings(GNANI_API_KEY="test_key_factory", SPEECH_TRANSCRIPTION_PROVIDER="gnani")
    def test_default_provider_returns_gnani_client(self):
        client = get_stt_client()
        self.assertIsInstance(client, GnaniPrismaBatchClient)
        self.assertIsInstance(client, BaseSpeechToTextClient)

    @override_settings(GNANI_API_KEY="test_key_factory", SPEECH_TRANSCRIPTION_PROVIDER="GNANI")
    def test_factory_is_case_insensitive(self):
        client = get_stt_client()
        self.assertIsInstance(client, GnaniPrismaBatchClient)

    @override_settings(GNANI_API_KEY="test_key_factory")
    def test_explicit_provider_parameter_overrides_setting(self):
        client = get_stt_client(provider_name="gnani")
        self.assertIsInstance(client, GnaniPrismaBatchClient)

    def test_unsupported_provider_raises_configuration_error(self):
        with self.assertRaises(STTConfigurationError) as ctx:
            get_stt_client(provider_name="unsupported_provider")
        self.assertIn("Unsupported speech transcription provider", str(ctx.exception))

    def test_google_provider_selection_returns_google_speech_client(self):
        with patch("speech_analytics.services.google_speech.speech_v1.SpeechClient"):
            client = get_stt_client(provider_name="google")
            self.assertIsInstance(client, BaseSpeechToTextClient)
            from speech_analytics.services.google_speech import GoogleSpeechClient
            self.assertIsInstance(client, GoogleSpeechClient)
