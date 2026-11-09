"""Custom exception hierarchy for Gnani Prisma Speech-to-Text client.
Ensures errors never leak sensitive API credentials or authorization headers.
"""


class GnaniSTTError(Exception):
    """Base exception for all Gnani STT client errors."""

    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}


class GnaniConfigurationError(GnaniSTTError):
    """Raised when required API credentials or base settings are missing."""
    pass


class GnaniAuthenticationError(GnaniSTTError):
    """Raised when the API key is missing, invalid, or unauthorized (HTTP 401/403)."""
    pass


class GnaniRateLimitError(GnaniSTTError):
    """Raised when rate limits or concurrency quotas are reached (HTTP 429)."""
    pass


class GnaniAPIResponseError(GnaniSTTError):
    """Raised when Gnani returns a 4xx or 5xx server/client error."""
    pass


class GnaniConnectionError(GnaniSTTError):
    """Raised when a network failure or connection dropped occurs."""
    pass


class GnaniTimeoutError(GnaniSTTError):
    """Raised when an HTTP request times out."""
    pass


class GnaniMalformedResponseError(GnaniSTTError):
    """Raised when the API returns an unexpected or malformed payload."""
    pass
