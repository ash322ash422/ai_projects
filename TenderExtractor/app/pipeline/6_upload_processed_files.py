from pathlib import Path
import sys

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

# Make config importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    AZURE_STORAGE_CONNECTION_STRING,
    BLOB_CONTAINER_NAME,
    PROCESSED_CONTAINER,
    PROJECT_ROOT
)


PDF_NAME = "01_tender_mini_version.pdf"
BASE_NAME = Path(PDF_NAME).stem

CLEAN_EXCEL = (
    PROJECT_ROOT
    / "cache"
    / f"{BASE_NAME}_CLEAN.xlsx"
)

def get_container_client(container_name):
    """
    Get a container. Create it if it does not exist.
    """

    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )

    try:
        blob_service_client.create_container(container_name)
        print(f"Created container: {container_name}")

    except ResourceExistsError:
        print(f"Container exists: {container_name}")

    return blob_service_client.get_container_client(container_name)


# -------------------------------------------------------------------

def upload_file(
    container_client,
    local_file,
    blob_name=None
):
    """
    Upload a local file.
    """

    local_file = Path(local_file)

    if not local_file.exists():
        raise FileNotFoundError(local_file)

    if blob_name is None:
        blob_name = local_file.name

    with open(local_file, "rb") as f:

        container_client.upload_blob(
            name=blob_name,
            data=f,
            overwrite=True
        )

    print(f"Uploaded {blob_name}")


# -------------------------------------------------------------------

def move_blob(
    blob_service_client,
    source_container,
    destination_container,
    blob_name
):
    """
    Move blob between containers.
    """

    source_blob = blob_service_client.get_blob_client(
        source_container,
        blob_name
    )

    destination_blob = blob_service_client.get_blob_client(
        destination_container,
        blob_name
    )

    print(f"Copying {blob_name}...")

    destination_blob.start_copy_from_url(
        source_blob.url
    )

    print("Deleting original...")

    source_blob.delete_blob()

    print("Move complete.")


# -------------------------------------------------------------------

def list_blobs(container_client):

    print("\nContainer Contents")
    print("-" * 40)

    count = 0

    for blob in container_client.list_blobs():

        count += 1
        print(blob.name)

    if count == 0:
        print("(empty)")


# -------------------------------------------------------------------

if __name__ == "__main__":

    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )

    processed_client = get_container_client(
        PROCESSED_CONTAINER
    )

    print("\nUploading Excel...")

    upload_file(
        processed_client,
        CLEAN_EXCEL
    )

    print("\nMoving PDF...")

    move_blob(
        blob_service_client,
        source_container=BLOB_CONTAINER_NAME,
        destination_container=PROCESSED_CONTAINER,
        blob_name=PDF_NAME
    )

    print("\nProcessed Container")

    list_blobs(processed_client)

    print("\nIncoming Container")
    incoming_client = get_container_client(BLOB_CONTAINER_NAME)

    list_blobs(incoming_client)