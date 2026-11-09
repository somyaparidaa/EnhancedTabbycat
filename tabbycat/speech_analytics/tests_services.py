import io
import json
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings
import requests

from speech_analytics.services.exceptions import (
    GnaniAPIResponseError,
    GnaniAuthenticationError,
    GnaniConfigurationError,
    GnaniConnectionError,
    GnaniMalformedResponseError,
    GnaniRateLimitError,
    GnaniTimeoutError,
)
from speech_analytics.services.gnani import GnaniPrismaBatchClient


@override_settings(GNANI_API_KEY="test-mock-api-key")
class GnaniPrismaBatchClientTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        self.client = GnaniPrismaBatchClient(api_key="test-mock-api-key", base_url="https://mock.gnani.test")
        self.dummy_audio = io.BytesIO(b"RIFFdummywavdata")
        self.dummy_audio.name = "test_speech.wav"

    def _mock_response(self, status_code=200, json_data=None, text="", headers=None):
        response = MagicMock(spec=requests.Response)
        response.status_code = status_code
        response.text = text or (json.dumps(json_data) if json_data is not None else "")
        response.headers = headers or {}
        if json_data is not None:
            response.json.return_value = json_data
        else:
            response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        return response

    # 1. Successful job submission
    @patch.object(requests.Session, 'request')
    def test_successful_job_submission(self, mock_request):
        mock_request.return_value = self._mock_response(
            status_code=201,
            json_data={"job_id": "job_12345", "status": "CREATED"},
        )
        result = self.client.create_batch_job(
            audio_file=self.dummy_audio,
            language_code="en-IN",
            model="prisma-v2.5",
        )
        self.assertEqual(result['job_id'], "job_12345")
        self.assertEqual(result['status'], "CREATED")
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        self.assertEqual(kwargs['method'], 'POST')
        self.assertEqual(kwargs['url'], 'https://mock.gnani.test/stt/v3/batch/jobs')
        self.assertEqual(kwargs['headers']['X-API-Key-ID'], "test-mock-api-key")

    # 2. Successful job start
    @patch.object(requests.Session, 'request')
    def test_successful_job_start(self, mock_request):
        mock_request.return_value = self._mock_response(
            status_code=202,
            json_data={"job_id": "job_12345", "status": "STARTING"},
        )
        result = self.client.start_batch_job("job_12345")
        self.assertEqual(result['job_id'], "job_12345")
        self.assertEqual(result['status'], "STARTING")
        mock_request.assert_called_once()
        self.assertEqual(mock_request.call_args[1]['url'], 'https://mock.gnani.test/stt/v3/batch/jobs/job_12345/start')

    # 3. Processing status
    @patch.object(requests.Session, 'request')
    def test_processing_status(self, mock_request):
        mock_request.return_value = self._mock_response(
            status_code=200,
            json_data={"job_id": "job_12345", "status": "IN_PROGRESS", "progress": 45},
        )
        result = self.client.get_job_status("job_12345")
        self.assertEqual(result['status'], "IN_PROGRESS")
        self.assertEqual(result['progress'], 45)

    # 4. Completed job and transcript retrieval
    @patch.object(requests.Session, 'request')
    def test_completed_job_and_transcript_retrieval(self, mock_request):
        files_response = self._mock_response(
            status_code=200,
            json_data=[{"transcript_url": "https://storage.googleapis.com/gnani-transcripts/t1.json"}],
        )
        transcript_response = self._mock_response(
            status_code=200,
            json_data={
                "full_transcript": "We propose a policy for debate governance.",
                "language_code": "en-IN",
                "audio_duration": 420.5,
                "confidence": 0.96,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 3.5,
                        "text": "We propose a policy",
                        "speaker": "Speaker 1",
                        "confidence": 0.98,
                        "words": [
                            {"word": "We", "start": 0.0, "end": 0.5, "confidence": 0.99},
                            {"word": "propose", "start": 0.6, "end": 1.2, "confidence": 0.97},
                        ],
                    }
                ],
            },
        )
        mock_request.side_effect = [files_response, transcript_response]

        transcript = self.client.fetch_completed_transcript("job_12345")
        self.assertEqual(transcript['job_id'], "job_12345")
        self.assertEqual(transcript['text'], "We propose a policy for debate governance.")
        self.assertEqual(transcript['language_code'], "en-IN")
        self.assertEqual(transcript['duration'], 420.5)
        self.assertEqual(transcript['overall_confidence'], 0.96)
        self.assertEqual(len(transcript['segments']), 1)
        self.assertEqual(transcript['segments'][0]['speaker'], "Speaker 1")
        self.assertEqual(len(transcript['segments'][0]['words']), 2)

        # Verify X-API-Key-ID was NOT sent to external storage host
        download_call_kwargs = mock_request.call_args_list[1][1]
        self.assertNotIn('X-API-Key-ID', download_call_kwargs['headers'])

    # 5. Failed job
    @patch.object(requests.Session, 'request')
    def test_failed_job_status(self, mock_request):
        mock_request.return_value = self._mock_response(
            status_code=200,
            json_data={"job_id": "job_12345", "status": "FAILED", "error": "Corrupted audio header"},
        )
        result = self.client.get_job_status("job_12345")
        self.assertEqual(result['status'], "FAILED")
        self.assertEqual(result['error'], "Corrupted audio header")

    # 6. HTTP authentication failure
    @patch.object(requests.Session, 'request')
    def test_authentication_failure(self, mock_request):
        mock_request.return_value = self._mock_response(
            status_code=401,
            json_data={"message": "Invalid API Key"},
        )
        with self.assertRaises(GnaniAuthenticationError) as ctx:
            self.client.create_batch_job(self.dummy_audio)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertNotIn("test-mock-api-key", str(ctx.exception))

    # 7. Rate limiting (429)
    @patch.object(requests.Session, 'request')
    def test_rate_limiting(self, mock_request):
        mock_request.return_value = self._mock_response(
            status_code=429,
            json_data={"message": "Concurrency limit exceeded"},
        )
        with self.assertRaises(GnaniRateLimitError) as ctx:
            self.client.get_job_status("job_123")
        self.assertEqual(ctx.exception.status_code, 429)

    # 8. Server errors (500)
    @patch.object(requests.Session, 'request')
    def test_server_error(self, mock_request):
        mock_request.return_value = self._mock_response(
            status_code=500,
            json_data={"error": "Internal Server Error"},
        )
        with self.assertRaises(GnaniAPIResponseError) as ctx:
            self.client.get_job_status("job_123")
        self.assertEqual(ctx.exception.status_code, 500)

    # 9. Connection timeout
    @patch.object(requests.Session, 'request')
    def test_connection_timeout(self, mock_request):
        mock_request.side_effect = requests.exceptions.Timeout("Connection timed out")
        with self.assertRaises(GnaniTimeoutError):
            self.client.get_job_status("job_123")

    # 10. Malformed or unexpected responses
    @patch.object(requests.Session, 'request')
    def test_malformed_response(self, mock_request):
        mock_request.return_value = self._mock_response(
            status_code=200,
            text="<html>502 Bad Gateway</html>",
        )
        with self.assertRaises(GnaniMalformedResponseError):
            self.client.get_job_status("job_123")

    # 11. Missing API credentials
    def test_missing_api_credentials_raises_configuration_error(self):
        with override_settings(GNANI_API_KEY=None):
            with self.assertRaises(GnaniConfigurationError):
                GnaniPrismaBatchClient(api_key=None)

    # 12. Metadata sanitisation preserves legitimate keyterms
    def test_metadata_sanitisation_preserves_keyterms(self):
        dirty_payload = {
            "token": "secret-token-xyz",
            "api_key": "raw-key-12345",
            "access_token": "secret-access-token",
            "authorization": "Bearer xyz",
            "password": "pass",
            "safe_field": "valid_value",
            "keyterms": ["parliamentary", "motion", "prime minister"],
            "keyword_score": 0.95,
            "nested": {
                "secret": "nested-secret",
                "normal_data": 42,
                "keyterms_count": 3,
            },
        }
        sanitized = GnaniPrismaBatchClient.sanitize_metadata(dirty_payload)
        self.assertNotIn("token", sanitized)
        self.assertNotIn("api_key", sanitized)
        self.assertNotIn("access_token", sanitized)
        self.assertNotIn("authorization", sanitized)
        self.assertNotIn("password", sanitized)
        self.assertNotIn("secret", sanitized["nested"])

        # Legitimate debate STT keywords are preserved
        self.assertEqual(sanitized["keyterms"], ["parliamentary", "motion", "prime minister"])
        self.assertEqual(sanitized["keyword_score"], 0.95)
        self.assertEqual(sanitized["nested"]["keyterms_count"], 3)
        self.assertEqual(sanitized["nested"]["normal_data"], 42)

    # 13. Security: Insecure transcript_url rejected (HTTP vs HTTPS)
    def test_insecure_transcript_url_rejected(self):
        with self.assertRaises(GnaniAPIResponseError) as ctx:
            self.client.download_transcript("http://insecure.gnani.test/t1.json")
        self.assertIn("HTTPS is required", str(ctx.exception))

    # 14. File upload: Seekable stream is reset to 0
    @patch.object(requests.Session, 'request')
    def test_file_upload_stream_resets_seek(self, mock_request):
        mock_request.return_value = self._mock_response(
            status_code=201,
            json_data={"job_id": "job_seek_test", "status": "CREATED"},
        )
        stream = io.BytesIO(b"audio bytes content")
        stream.seek(10)  # Move pointer away from start
        self.assertEqual(stream.tell(), 10)

        self.client.create_batch_job(
            audio_file=("speech.wav", stream, "audio/wav"),
        )
        self.assertEqual(stream.tell(), 0)  # Reset to position 0 before upload
