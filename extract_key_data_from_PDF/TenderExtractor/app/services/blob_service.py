"""
Wraps Azure Blob Storage for uploads and outputs.

For the POC, if AZURE_STORAGE_CONNECTION_STRING is not configured, this
falls back to writing to local disk (uploads/ and output/) so the whole
pipeline can be demoed without an Azure subscription. Swapping in real
Blob Storage later only requires setting the connection string in .env.
"""

from pathlib import Path
from typing import Optional

from app.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
except ImportError:  # SDK optional for local-only POC runs
    BlobServiceClient = None
    ContentSettings = None


class BlobService:
    def __init__(self) -> None:
        self.use_azure = bool(settings.AZURE_STORAGE_CONNECTION_STRING) and BlobServiceClient is not None
        if self.use_azure:
            self.client = BlobServiceClient.from_connection_string(
                settings.AZURE_STORAGE_CONNECTION_STRING
            )
            self._ensure_container(settings.AZURE_STORAGE_CONTAINER_UPLOADS)
            self._ensure_container(settings.AZURE_STORAGE_CONTAINER_OUTPUT)
        else:
            logger.warning(
                "AZURE_STORAGE_CONNECTION_STRING not set - using local disk "
                "fallback for blob storage (fine for a POC demo)."
            )
            Path(settings.LOCAL_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
            Path(settings.LOCAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def _ensure_container(self, name: str) -> None:
        try:
            self.client.create_container(name)
        except Exception:
            pass  # already exists

    # ---------------------------------------------------------------
    def upload_input_file(self, filename: str, data: bytes) -> str:
        """Stores the uploaded PDF. Returns a reference (path or blob name)."""
        if self.use_azure:
            blob = self.client.get_blob_client(
                container=settings.AZURE_STORAGE_CONTAINER_UPLOADS, blob=filename
            )
            blob.upload_blob(data, overwrite=True)
            return f"{settings.AZURE_STORAGE_CONTAINER_UPLOADS}/{filename}"

        path = Path(settings.LOCAL_UPLOAD_DIR) / filename
        path.write_bytes(data)
        return str(path)

    def upload_output_file(self, filename: str, data: bytes) -> str:
        """Stores the generated Excel file. Returns a reference used for download."""
        if self.use_azure:
            blob = self.client.get_blob_client(
                container=settings.AZURE_STORAGE_CONTAINER_OUTPUT, blob=filename
            )
            content_settings = ContentSettings(
                content_type=(
                    "application/vnd.openxmlformats-officedocument"
                    ".spreadsheetml.sheet"
                )
            )
            blob.upload_blob(data, overwrite=True, content_settings=content_settings)
            return f"{settings.AZURE_STORAGE_CONTAINER_OUTPUT}/{filename}"

        path = Path(settings.LOCAL_OUTPUT_DIR) / filename
        path.write_bytes(data)
        return str(path)

    def read_output_file(self, filename: str) -> Optional[bytes]:
        if self.use_azure:
            blob = self.client.get_blob_client(
                container=settings.AZURE_STORAGE_CONTAINER_OUTPUT, blob=filename
            )
            if not blob.exists():
                return None
            return blob.download_blob().readall()

        path = Path(settings.LOCAL_OUTPUT_DIR) / filename
        if not path.exists():
            return None
        return path.read_bytes()


blob_service = BlobService()
