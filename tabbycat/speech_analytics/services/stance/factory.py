"""Factory and caching registry for debate stance classifiers.

Maintains thread-safe singleton model caching across Django web requests
to avoid reloading multi-hundred-megabyte weights on every invocation.
"""

import logging
import os
from typing import Optional

from django.conf import settings

from .base import BaseStanceClassifier
from .exceptions import StanceConfigurationError
from .mock import MockStanceClassifier

logger = logging.getLogger(__name__)

# Module-level singleton instance cache
_CACHED_STANCE_CLASSIFIER: Optional[BaseStanceClassifier] = None


def get_stance_classifier(
    backend: Optional[str] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    force_reload: bool = False,
) -> BaseStanceClassifier:
    """Return the configured stance classification service instance.

    Args:
        backend: Provider backend ('deberta' or 'mock').
            Defaults to settings.STANCE_CLASSIFIER_BACKEND or 'deberta'.
        model_path: Path to local model checkpoint directory.
            Defaults to settings.STANCE_MODEL_PATH.
        device: Compute device ('cpu', 'mps', 'cuda').
            Defaults to settings.STANCE_CLASSIFIER_DEVICE.
        force_reload: When True, bypasses and resets the singleton cache.

    Returns:
        An instance conforming to BaseStanceClassifier.
    """
    global _CACHED_STANCE_CLASSIFIER

    if _CACHED_STANCE_CLASSIFIER is not None and not force_reload:
        return _CACHED_STANCE_CLASSIFIER

    selected_backend = (backend or getattr(settings, 'STANCE_CLASSIFIER_BACKEND', 'deberta')).lower().strip()

    if selected_backend == 'mock':
        logger.info("Instantiating MockStanceClassifier")
        classifier = MockStanceClassifier()
        _CACHED_STANCE_CLASSIFIER = classifier
        return classifier

    elif selected_backend == 'deberta':
        from .deberta import DeBERTaStanceClassifier

        configured_path = model_path or getattr(settings, 'STANCE_MODEL_PATH', '/Users/somya/Developer/stance_ft')
        if not configured_path or not os.path.exists(configured_path):
            raise StanceConfigurationError(
                f"Configured STANCE_MODEL_PATH does not exist at '{configured_path}'."
            )

        configured_device = device or getattr(settings, 'STANCE_CLASSIFIER_DEVICE', None)
        conf_thresh = float(getattr(settings, 'STANCE_CONFIDENCE_THRESHOLD', 0.55))
        margin_thresh = float(getattr(settings, 'STANCE_MARGIN_THRESHOLD', 0.15))

        logger.info("Initializing DeBERTaStanceClassifier from %s", configured_path)
        classifier = DeBERTaStanceClassifier(
            model_path=configured_path,
            device=configured_device,
            confidence_threshold=conf_thresh,
            margin_threshold=margin_thresh,
        )
        _CACHED_STANCE_CLASSIFIER = classifier
        return classifier

    else:
        raise StanceConfigurationError(
            f"Unsupported stance classification backend '{selected_backend}'. Valid options: 'deberta', 'mock'."
        )


def reset_stance_classifier_cache():
    """Clear cached model instance from memory."""
    global _CACHED_STANCE_CLASSIFIER
    _CACHED_STANCE_CLASSIFIER = None
