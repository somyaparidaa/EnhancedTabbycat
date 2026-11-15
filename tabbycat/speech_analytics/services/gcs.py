"""Google Cloud Storage service adapter using the official google-cloud-storage SDK.
Handles secure upload and cleanup of audio files for Google Cloud Speech long-running recognition.
Authenticates seamlessly using Google Application Default Credentials (ADC).
Never modifies or deletes local user recordings.
"""

import logging
import os
import uuid
from typing import Optional

from django.conf import settings
from google.auth.exceptions import DefaultCredentialsError, GoogleAuthError
from google.cloud import storage

from .base import (
    STTAuthenticationError,
    STTConfigurationError,
    STTConnectionError,
    STTError,
)

logger = logging.getLogger(__name__)


class GoogleCloudStorageService:
    """Manages audio object lifecycle in Google Cloud Storage for long audio transcription.

    Features:
    - Uses official `google.cloud.storage.Client` with ADC.
    - Uses configurable bucket name (`settings.GOOGLE_CLOUD_STORAGE_BUCKET_NAME`).
    - Creates unique, collision-resistant object paths (`speech_recordings/<uuid>/<filename>`).
    - Cleans up temporary GCS audio objects only upon explicit request or safe completion.
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        client: Optional[storage.Client] = None,
    ):
        self.bucket_name = bucket_name or getattr(settings, 'GOOGLE_CLOUD_STORAGE_BUCKET_NAME', '')

        if client:
            self._client = client
        else:
            try:
                self._client = storage.Client()
            except (DefaultCredentialsError, GoogleAuthError) as exc:
                logger.error("GCS authentication failed using ADC: %s", exc)
                raise STTAuthenticationError(
                    f"Google Cloud Storage authentication failed using ADC: {exc}"
                ) from exc
            except Exception as exc:
                logger.error("Failed to initialize Google Cloud Storage client: %s", exc)
                raise STTConfigurationError(
                    f"Failed to initialize Google Cloud Storage client: {exc}"
                ) from exc

    def _get_bucket(self) -> storage.Bucket:
        if not self.bucket_name:
            raise STTConfigurationError(
                "Google Cloud Storage bucket name is not configured. "
                "Please set GOOGLE_CLOUD_STORAGE_BUCKET_NAME in settings or environment."
            )
        try:
            return self._client.bucket(self.bucket_name)
        except Exception as exc:
            raise STTConnectionError(f"Failed to access GCS bucket '{self.bucket_name}': {exc}") from exc

    def upload_audio_bytes(
        self,
        audio_content: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> str:
        """Upload audio bytes to a unique GCS object path.

        Returns:
            GCS URI in the format 'gs://<bucket_name>/<object_name>'
        """
        bucket = self._get_bucket()
        unique_id = uuid.uuid4().hex
        clean_filename = os.path.basename(filename)
        blob_path = f"speech_recordings/{unique_id}/{clean_filename}"

        try:
            blob = bucket.blob(blob_path)
            blob.upload_from_string(audio_content, content_type=content_type)
            gcs_uri = f"gs://{self.bucket_name}/{blob_path}"
            logger.info("Successfully uploaded %d bytes to GCS: %s", len(audio_content), gcs_uri)
            return gcs_uri
        except Exception as exc:
            logger.exception("Failed to upload audio to GCS bucket '%s': %s", self.bucket_name, exc)
            raise STTConnectionError(f"Failed to upload audio to Google Cloud Storage: {exc}") from exc

    def delete_audio_by_uri(self, gcs_uri: str) -> bool:
        """Safely delete an audio object from GCS given its gs:// URI.

        Never touches any local user recordings.
        """
        if not gcs_uri or not gcs_uri.startswith("gs://"):
            logger.warning("Invalid GCS URI provided for deletion: %s", gcs_uri)
            return False

        try:
            # Parse gs://<bucket>/<blob_name>
            parts = gcs_uri[5:].split("/", 1)
            if len(parts) != 2:
                logger.warning("Malformed GCS URI format: %s", gcs_uri)
                return False

            bucket_name, blob_name = parts
            bucket = self._client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            if blob.exists():
                blob.delete()
                logger.info("Successfully deleted temporary GCS audio object: %s", gcs_uri)
                return True
            else:
                logger.info("GCS audio object does not exist or was already deleted: %s", gcs_uri)
                return True
        except Exception as exc:
            logger.warning("Failed to delete GCS object '%s': %s", gcs_uri, exc)
            return False
