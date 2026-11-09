import json
import logging
import mimetypes
import os
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from django.conf import settings
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import (
    GnaniAPIResponseError,
    GnaniAuthenticationError,
    GnaniConfigurationError,
    GnaniConnectionError,
    GnaniMalformedResponseError,
    GnaniRateLimitError,
    GnaniTimeoutError,
)

logger = logging.getLogger(__name__)

# Sensitive keys to strip from metadata payloads
SENSITIVE_KEY_PATTERNS = {
    'api_key',
    'apikey',
    'x-api-key-id',
    'access_token',
    'refresh_token',
    'token',
    'authorization',
    'secret',
    'client_secret',
    'password',
    'auth_header',
}


class GnaniPrismaBatchClient:
    """Service client for Gnani Prisma v2.5 / v3.0 Batch Speech-to-Text API.
    Operates strictly server-side using configured credentials.
    """

    DEFAULT_BASE_URL = "https://api.vachana.ai"
    DEFAULT_TIMEOUT: Tuple[int, int] = (10, 60)  # (connect_timeout, read_timeout)

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[Tuple[int, int]] = None,
    ):
        configured_key = api_key or getattr(settings, 'GNANI_API_KEY', None)
        if not configured_key or not str(configured_key).strip():
            raise GnaniConfigurationError(
                "Gnani API Key is not configured. Please set GNANI_API_KEY in your environment or local settings."
            )

        self._api_key = str(configured_key).strip()
        self.base_url = (base_url or getattr(settings, 'GNANI_API_BASE_URL', self.DEFAULT_BASE_URL)).rstrip('/')
        self.timeout = timeout or getattr(settings, 'GNANI_API_TIMEOUT', self.DEFAULT_TIMEOUT)

        self.session = requests.Session()

        # Mount safe retries ONLY for idempotent read-only requests (GET/HEAD)
        # Note: POST is explicitly NOT in method_whitelist to avoid duplicate jobs and billing.
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
            allowed_methods=frozenset(['GET', 'HEAD', 'OPTIONS']),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get_headers(self) -> Dict[str, str]:
        """Return authentication and common headers.
        Uses X-API-Key-ID as specified in Gnani AI Batch documentation.
        """
        return {
            'X-API-Key-ID': self._api_key,
        }

    def _sanitize_log_url(self, url: str) -> str:
        return url.split('?')[0]

    def _request(
        self,
        method: str,
        path_or_url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        include_auth: bool = True,
        allow_redirects: bool = True,
    ) -> requests.Response:
        """Execute an HTTP request with safe error handling and timeout protection."""
        url = path_or_url if path_or_url.startswith('http') else f"{self.base_url}{path_or_url}"

        req_headers = {}
        if include_auth:
            req_headers.update(self._get_headers())
        if headers:
            req_headers.update(headers)

        safe_url = self._sanitize_log_url(url)
        logger.debug("Executing Gnani API %s request to %s", method, safe_url)

        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=req_headers,
                data=data,
                files=files,
                params=params,
                timeout=self.timeout,
                allow_redirects=allow_redirects,
            )
        except requests.exceptions.Timeout as exc:
            logger.error("Timeout connecting to Gnani API at %s", safe_url)
            raise GnaniTimeoutError(f"Gnani API request timed out ({safe_url}).") from exc
        except requests.exceptions.ConnectionError as exc:
            logger.error("Connection error while connecting to Gnani API at %s", safe_url)
            raise GnaniConnectionError(f"Failed to connect to Gnani API ({safe_url}).") from exc
        except requests.exceptions.RequestException as exc:
            logger.error("Unexpected network error while contacting Gnani API at %s", safe_url)
            raise GnaniConnectionError(f"Network error contacting Gnani API: {str(exc)}") from exc

        return self._handle_response_status(response, safe_url)

    def _handle_response_status(self, response: requests.Response, safe_url: str) -> requests.Response:
        """Inspect HTTP status codes and raise descriptive domain exceptions without leaking secrets."""
        status = response.status_code
        if 200 <= status < 300:
            return response

        try:
            error_payload = response.json()
        except Exception:
            error_payload = {'error': response.text}

        error_msg = (
            error_payload.get('message')
            or error_payload.get('error')
            or error_payload.get('detail')
            or f"HTTP error {status}"
        )

        if status in (401, 403):
            logger.warning("Gnani authentication failed at %s with status %d", safe_url, status)
            raise GnaniAuthenticationError(
                f"Authentication with Gnani API failed ({status}). Check your GNANI_API_KEY.",
                status_code=status,
                response_data=error_payload,
            )
        elif status == 429:
            logger.warning("Gnani API rate limit encountered at %s", safe_url)
            raise GnaniRateLimitError(
                "Gnani API rate limit reached. Please retry later.",
                status_code=status,
                response_data=error_payload,
            )
        else:
            logger.warning("Gnani API error response %d from %s", status, safe_url)
            raise GnaniAPIResponseError(
                f"Gnani API error ({status}): {error_msg}",
                status_code=status,
                response_data=error_payload,
            )

    def create_batch_job(
        self,
        audio_file: Union[BinaryIO, Tuple[str, BinaryIO, str]],
        language_code: str = "en-IN",
        model: str = "prisma-v2.5",
        with_diarization: bool = False,
        mode: str = "transcribe",
        with_denoise: bool = False,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit an audio file and create an asynchronous batch transcription job.
        Note: This is never automatically retried to avoid duplicate jobs and billing.
        Endpoint: POST /stt/v3/batch/jobs
        """
        config_payload = {
            "model": model,
            "language_code": language_code,
            "mode": mode,
            "with_diarization": with_diarization,
            "with_denoise": with_denoise,
            "is_multi_channel": False,
        }
        if extra_config:
            config_payload.update(extra_config)

        if isinstance(audio_file, tuple):
            filename, file_stream, content_type = audio_file
            if hasattr(file_stream, 'seek') and callable(file_stream.seek):
                file_stream.seek(0)
            files = [
                ('config', (None, json.dumps(config_payload), 'application/json')),
                ('files', (filename, file_stream, content_type)),
            ]
        else:
            if hasattr(audio_file, 'seek') and callable(audio_file.seek):
                audio_file.seek(0)
            filename = getattr(audio_file, 'name', 'audio.wav')
            basename = os.path.basename(filename) if filename else 'audio.wav'
            guessed_type, _ = mimetypes.guess_type(basename)
            content_type = guessed_type or 'audio/wav'
            files = [
                ('config', (None, json.dumps(config_payload), 'application/json')),
                ('files', (basename, audio_file, content_type)),
            ]

        response = self._request('POST', '/stt/v3/batch/jobs', files=files, include_auth=True)
        try:
            data = response.json()
        except Exception as exc:
            raise GnaniMalformedResponseError("Gnani API returned non-JSON response on job creation.") from exc

        job_id = data.get('job_id') or data.get('id') or data.get('data', {}).get('job_id')
        if not job_id:
            raise GnaniMalformedResponseError(
                f"Job creation succeeded with status {response.status_code} but no job_id was found in response.",
                response_data=data,
            )

        status = data.get('status') or data.get('data', {}).get('status', 'CREATED')
        return {
            'job_id': str(job_id),
            'status': status,
            'raw_summary': self.sanitize_metadata(data),
        }

    def start_batch_job(self, job_id: str) -> Dict[str, Any]:
        """Trigger execution of an already created batch job.
        Endpoint: POST /stt/v3/batch/jobs/{job_id}/start
        """
        if not job_id:
            raise ValueError("job_id is required to start a batch job.")

        response = self._request('POST', f'/stt/v3/batch/jobs/{job_id}/start', include_auth=True)
        try:
            data = response.json()
        except Exception as exc:
            raise GnaniMalformedResponseError("Gnani API returned non-JSON response on job start.") from exc

        status = data.get('status') or data.get('data', {}).get('status', 'STARTING')
        return {
            'job_id': job_id,
            'status': status,
            'raw_summary': self.sanitize_metadata(data),
        }

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Poll the current processing status of a batch job.
        Endpoint: GET /stt/v3/batch/jobs/{job_id}
        """
        if not job_id:
            raise ValueError("job_id is required to query job status.")

        response = self._request('GET', f'/stt/v3/batch/jobs/{job_id}', include_auth=True)
        try:
            data = response.json()
        except Exception as exc:
            raise GnaniMalformedResponseError("Gnani API returned non-JSON response on job status poll.") from exc

        status = data.get('status') or data.get('data', {}).get('status', 'UNKNOWN')
        return {
            'job_id': job_id,
            'status': status,
            'progress': data.get('progress') or data.get('data', {}).get('progress'),
            'error': data.get('error') or data.get('error_message') or data.get('data', {}).get('error'),
            'raw_summary': self.sanitize_metadata(data),
        }

    def get_job_files(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieve completed file URLs for a batch job.
        Endpoint: GET /stt/v3/batch/jobs/{job_id}/files
        """
        if not job_id:
            raise ValueError("job_id is required to fetch job files.")

        response = self._request('GET', f'/stt/v3/batch/jobs/{job_id}/files', include_auth=True)
        try:
            data = response.json()
        except Exception as exc:
            raise GnaniMalformedResponseError("Gnani API returned non-JSON response on job files request.") from exc

        files_list = data if isinstance(data, list) else (data.get('files') or data.get('data') or [])
        return [self.sanitize_metadata(f) for f in files_list if isinstance(f, dict)]

    def download_transcript(self, transcript_url: str) -> Dict[str, Any]:
        """Download transcript JSON from the provided transcript URL and normalize it into an internal structure.
        Security requirements:
        1. Validate HTTPS scheme.
        2. Never forward X-API-Key-ID to external / third-party storage hosts.
        3. Prevent SSRF / internal IP redirection.
        """
        if not transcript_url:
            raise ValueError("transcript_url is required to download transcript.")

        parsed = urlparse(transcript_url)
        if parsed.scheme.lower() != 'https':
            raise GnaniAPIResponseError(
                f"Insecure transcript_url scheme '{parsed.scheme}'. HTTPS is required for transcript downloads."
            )

        # Host check: Do NOT send X-API-Key-ID to external presigned S3 / cloud storage buckets
        parsed_base = urlparse(self.base_url)
        is_same_host = (parsed.netloc.lower() == parsed_base.netloc.lower())
        include_auth = is_same_host

        response = self._request(
            'GET',
            transcript_url,
            include_auth=include_auth,
            allow_redirects=False,  # Prevent automatic credential/SSRF redirection
        )

        # Handle redirects manually if any, validating HTTPS and stripping auth
        if response.status_code in (301, 302, 303, 307, 308):
            redirect_url = response.headers.get('Location')
            if not redirect_url:
                raise GnaniAPIResponseError("Redirect received without Location header.")
            parsed_redirect = urlparse(redirect_url)
            if parsed_redirect.scheme.lower() != 'https':
                raise GnaniAPIResponseError(f"Insecure redirect scheme '{parsed_redirect.scheme}'. HTTPS is required.")
            response = self._request(
                'GET',
                redirect_url,
                include_auth=False,
                allow_redirects=True,
            )

        try:
            payload = response.json()
        except Exception as exc:
            raise GnaniMalformedResponseError("Failed to parse transcript payload as JSON.") from exc

        return self.normalize_transcript_payload(payload)

    def fetch_completed_transcript(self, job_id: str) -> Dict[str, Any]:
        """Convenience method: fetches job files and downloads the first transcript for the job."""
        files = self.get_job_files(job_id)
        if not files:
            raise GnaniMalformedResponseError(f"No output files found for completed job {job_id}.")

        first_file = files[0]
        transcript_url = first_file.get('transcript_url') or first_file.get('url')
        if not transcript_url:
            raise GnaniMalformedResponseError(
                f"No transcript_url found in job file metadata for job {job_id}.",
                response_data=first_file,
            )

        normalized = self.download_transcript(transcript_url)
        normalized['job_id'] = job_id
        return normalized

    def normalize_transcript_payload(self, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a predictable internal structure from the Gnani transcript payload.
        Tolerates documented variations across v2.5 and v3.0 schemas.
        """
        if not isinstance(raw_payload, dict):
            raise GnaniMalformedResponseError(f"Expected dict for transcript payload, got {type(raw_payload).__name__}.")

        full_text = (
            raw_payload.get('full_transcript')
            or raw_payload.get('transcript')
            or raw_payload.get('text')
            or ""
        )

        segments = raw_payload.get('segments') or raw_payload.get('sentences') or []
        normalized_segments = []

        if isinstance(segments, list):
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                normalized_segments.append({
                    'start': seg.get('start', 0.0),
                    'end': seg.get('end', 0.0),
                    'text': seg.get('text', ''),
                    'speaker': seg.get('speaker'),
                    'confidence': seg.get('confidence'),
                    'words': [
                        {
                            'word': w.get('word', ''),
                            'start': w.get('start', 0.0),
                            'end': w.get('end', 0.0),
                            'confidence': w.get('confidence'),
                        }
                        for w in seg.get('words', [])
                        if isinstance(w, dict)
                    ] if 'words' in seg and isinstance(seg['words'], list) else None,
                })

            if not full_text and normalized_segments:
                full_text = " ".join(s['text'] for s in normalized_segments if s.get('text'))

        language_code = (
            raw_payload.get('language_code')
            or raw_payload.get('language_detected')
            or raw_payload.get('language')
            or 'en'
        )

        duration = raw_payload.get('audio_duration') or raw_payload.get('duration')
        confidence = raw_payload.get('confidence')

        return {
            'text': full_text.strip(),
            'language_code': language_code,
            'duration': duration,
            'overall_confidence': confidence,
            'segments': normalized_segments,
            'metadata': self.sanitize_metadata(raw_payload),
        }

    @staticmethod
    def sanitize_metadata(data: Any) -> Any:
        """Strip sensitive authorization tokens, headers, and secret keys before storing or logging.
        Preserves legitimate STT fields such as 'keyterms', 'keyword_score', etc.
        """
        if isinstance(data, dict):
            sanitized = {}
            for k, v in data.items():
                lower_k = str(k).lower().strip()
                if lower_k in SENSITIVE_KEY_PATTERNS or lower_k.endswith('_token') or lower_k.endswith('_secret'):
                    continue
                sanitized[k] = GnaniPrismaBatchClient.sanitize_metadata(v)
            return sanitized
        elif isinstance(data, list):
            return [GnaniPrismaBatchClient.sanitize_metadata(item) for item in data]
        else:
            return data
