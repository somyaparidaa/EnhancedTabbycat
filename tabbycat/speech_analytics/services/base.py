"""Base interface and provider-neutral exception hierarchy for Speech-to-Text (STT) services.
Ensures view and business logic are fully decoupled from provider-specific APIs
(such as Gnani Prisma, Google Speech-to-Text, Whisper, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union


class STTError(Exception):
    """Base exception for all Speech-to-Text operations."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}


class STTConfigurationError(STTError):
    """Raised when provider configuration or credentials are missing or invalid."""
    pass


class STTAuthenticationError(STTError):
    """Raised when authentication with the STT provider fails (e.g. invalid key or credentials)."""
    pass


class STTRateLimitError(STTError):
    """Raised when rate limits or concurrency quotas are reached (e.g. HTTP 429)."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = 429,
        response_data: Optional[dict] = None,
        retry_after: Optional[int] = None,
    ):
        super().__init__(message, status_code=status_code, response_data=response_data)
        self.retry_after = retry_after


class STTAPIResponseError(STTError):
    """Raised when an STT provider returns an unhandled client or server error."""
    pass


class STTConnectionError(STTError):
    """Raised when network connectivity to the STT provider fails."""
    pass


class STTTimeoutError(STTError):
    """Raised when a request to the STT provider times out."""
    pass


class STTMalformedResponseError(STTError):
    """Raised when the STT provider returns an invalid or unexpected response structure."""
    pass


class BaseSpeechToTextClient(ABC):
    """Abstract base class establishing the contract for speech transcription providers.

    All providers (Gnani, Google, etc.) must implement this interface to support
    asynchronous debate speech transcription and status querying.
    """

    @abstractmethod
    def create_batch_job(
        self,
        audio_file: Union[BinaryIO, Tuple[str, BinaryIO, str]],
        language_code: str = "en-IN",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Submit an audio file and initiate an asynchronous transcription job.

        Args:
            audio_file: A binary file-like object or tuple of (filename, file_stream, content_type).
            language_code: BCP-47 / ISO language code (e.g. 'en-IN', 'en-US').
            **kwargs: Provider-specific configuration options.

        Returns:
            Dict containing at minimum:
                - 'job_id': str (the unique provider identifier for this asynchronous operation)
                - 'status': str (initial job status, e.g. 'CREATED', 'STARTING', 'PROCESSING')
                - 'raw_summary': Optional sanitized provider payload
        """
        pass

    @abstractmethod
    def start_batch_job(self, job_id: str) -> Dict[str, Any]:
        """Explicitly trigger execution of a created batch job if required by the provider.

        For providers where job creation immediately starts processing (e.g. Google Cloud STT),
        this can be a no-op that returns the current status.

        Args:
            job_id: The provider job identifier.

        Returns:
            Dict with at minimum 'job_id' and 'status'.
        """
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Poll the current processing status of an asynchronous batch job.

        Args:
            job_id: The provider job identifier.

        Returns:
            Dict containing at minimum:
                - 'job_id': str
                - 'status': str (e.g. 'COMPLETED', 'FAILED', 'IN_PROGRESS', 'PROCESSING', 'QUEUED')
                - 'progress': Optional[int/float]
                - 'error': Optional[str]
        """
        pass

    @abstractmethod
    def fetch_completed_transcript(self, job_id: str) -> Dict[str, Any]:
        """Retrieve and normalize the completed transcript and metadata for a job.

        Args:
            job_id: The provider job identifier.

        Returns:
            Dict containing normalized transcript fields:
                - 'text': str (full transcribed speech text)
                - 'language_code': str
                - 'duration': Optional[float] (audio duration in seconds)
                - 'overall_confidence': Optional[float]
                - 'segments': List[Dict] (diarized speaker segments with text, timestamps, speaker)
                - 'metadata': Dict (sanitized metadata dictionary for SpeechTranscript.metadata)
        """
        pass
