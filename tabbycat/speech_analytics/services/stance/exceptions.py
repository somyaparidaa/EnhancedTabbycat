"""Provider-neutral exceptions for debate stance classification services."""


class StanceError(Exception):
    """Base exception for stance classification errors."""

    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.message = message
        self.original_exception = original_exception


class StanceConfigurationError(StanceError):
    """Raised when model configuration, paths, or settings are invalid or missing."""
    pass


class StanceModelLoadError(StanceError):
    """Raised when model weights or tokenizer cannot be loaded into memory or device."""
    pass


class StanceInferenceError(StanceError):
    """Raised when an error occurs during sequence tokenization or model forward pass."""
    pass
