from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

from metadata import update_metadata

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (AZURE_STORAGE_CONNECTION_STRING, PROCESSING_METADATA, 
                    PROJECT_ROOT, 
                    URL_EXPIRY_HOURS,
                    PROCESSED_CONTAINER
)
                                 
PDF_NAME = "01_tender_mini_version.pdf"
BASE_NAME = Path(PDF_NAME).stem

EXCEL_CLEAN = (
    PROJECT_ROOT / "cache" / f"{BASE_NAME}_CLEAN.xlsx"
)

PDF_NAME = "01_tender_mini_version.pdf"
BASE_NAME = Path(PDF_NAME).stem

CLEAN_EXCEL = (
    PROJECT_ROOT
    / "cache"
    / f"{BASE_NAME}_CLEAN.xlsx"
)

from azure.storage.blob import (
    BlobServiceClient,
    generate_blob_sas,
    BlobSasPermissions,
)


def generate_download_url(
    connection_string: str,
    container_name: str,
    blob_name: str,
    expiry_hours: int = 24
):
    """
    Generate a read-only SAS URL for a blob.

    Parameters
    ----------
    connection_string : str
        Azure Storage connection string.

    container_name : str
        Name of the blob container.

    blob_name : str
        Blob filename.

    expiry_hours : int
        Number of hours the URL remains valid.

    Returns
    -------
    str
        Read-only SAS URL.
    """

    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string
    )

    account_name = blob_service_client.account_name

    account_key = (
        blob_service_client.credential.account_key
    )

    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=expiry_hours),
    )

    blob_url = (
        f"https://{account_name}.blob.core.windows.net/"
        f"{container_name}/{blob_name}"
    )

    return f"{blob_url}?{sas_token}"



if __name__ == "__main__":

    download_url = generate_download_url(
        connection_string=AZURE_STORAGE_CONNECTION_STRING,
        container_name=PROCESSED_CONTAINER,
        blob_name=EXCEL_CLEAN.name,
        expiry_hours=URL_EXPIRY_HOURS
    )

    print("\nDownload URL")
    print("=" * 80)
    print(download_url)

    update_metadata(
        metadata_file=PROCESSING_METADATA,
        pdf_name=PDF_NAME,
        excel_name=CLEAN_EXCEL.name,
        download_url=download_url,
        status="Completed"
    )