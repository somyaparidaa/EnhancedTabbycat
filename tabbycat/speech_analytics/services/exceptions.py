"""Exception hierarchy for Gnani Prisma Speech-to-Text client.
Inherits from provider-neutral STT exceptions to ensure backward compatibility
while allowing views to catch generic STTError.
"""

from .base import (
    STTAPIResponseError,
    STTAuthenticationError,
    STTConfigurationError,
    STTConnectionError,
    STTError,
    STTMalformedResponseError,
    STTRateLimitError,
    STTTimeoutError,
)


class GnaniSTTError(STTError):
    """Base exception for all Gnani STT client errors."""
    pass


class GnaniConfigurationError(GnaniSTTError, STTConfigurationError):
    """Raised when required Gnani API credentials or settings are missing."""
    pass


class GnaniAuthenticationError(GnaniSTTError, STTAuthenticationError):
    """Raised when the Gnani API key is missing, invalid, or unauthorized (HTTP 401/403)."""
    pass


class GnaniRateLimitError(GnaniSTTError, STTRateLimitError):
    """Raised when Gnani rate limits or concurrency quotas are reached (HTTP 429)."""
    pass


class GnaniAPIResponseError(GnaniSTTError, STTAPIResponseError):
    """Raised when Gnani returns a 4xx or 5xx server/client error."""
    pass


class GnaniConnectionError(GnaniSTTError, STTConnectionError):
    """Raised when a network failure or dropped connection occurs with Gnani API."""
    pass


class GnaniTimeoutError(GnaniSTTError, STTTimeoutError):
    """Raised when an HTTP request to Gnani times out."""
    pass


class GnaniMalformedResponseError(GnaniSTTError, STTMalformedResponseError):
    """Raised when Gnani returns an unexpected or malformed payload."""
    pass
