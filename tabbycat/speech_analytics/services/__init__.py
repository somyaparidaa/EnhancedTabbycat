"""Service layer for speech analytics and external speech-to-text integrations."""

from .base import (
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
from .factory import get_stt_client
from .gnani import GnaniPrismaBatchClient
from .google_speech import GoogleSpeechClient

__all__ = [
    'BaseSpeechToTextClient',
    'STTError',
    'STTConfigurationError',
    'STTAuthenticationError',
    'STTRateLimitError',
    'STTAPIResponseError',
    'STTConnectionError',
    'STTTimeoutError',
    'STTMalformedResponseError',
    'get_stt_client',
    'GnaniPrismaBatchClient',
    'GoogleSpeechClient',
]

