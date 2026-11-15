"""Factory for Speech-to-Text provider instantiation.
Reads `settings.SPEECH_TRANSCRIPTION_PROVIDER` to return the appropriate STT client.
"""

import logging
from typing import Optional

from django.conf import settings

from .base import BaseSpeechToTextClient, STTConfigurationError
from .gnani import GnaniPrismaBatchClient
from .google_speech import GoogleSpeechClient

logger = logging.getLogger(__name__)

# Registry of supported providers
PROVIDER_REGISTRY = {
    'gnani': GnaniPrismaBatchClient,
    'google': GoogleSpeechClient,
}


def get_stt_client(provider_name: Optional[str] = None) -> BaseSpeechToTextClient:
    """Return an instantiated Speech-to-Text client conforming to BaseSpeechToTextClient.

    Args:
        provider_name: Optional explicit provider name ('gnani', 'google', etc.).
            If omitted, defaults to settings.SPEECH_TRANSCRIPTION_PROVIDER.

    Returns:
        An instance of a BaseSpeechToTextClient subclass.

    Raises:
        STTConfigurationError: If the specified provider is unknown or not supported.
    """
    provider = (
        provider_name
        or getattr(settings, 'SPEECH_TRANSCRIPTION_PROVIDER', 'gnani')
        or 'gnani'
    ).strip().lower()

    client_cls = PROVIDER_REGISTRY.get(provider)
    if not client_cls:
        valid_providers = list(PROVIDER_REGISTRY.keys())
        raise STTConfigurationError(
            f"Unsupported speech transcription provider '{provider}'. "
            f"Supported providers: {', '.join(valid_providers)}."
        )

    return client_cls()
