from pathlib import Path
import sys

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

# Add the 'app' directory to the system path so 'config' can be found
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER_NAME, LOCAL_PDF_DIRECTORY

# Create/Get Container
def get_container_client(container_name: str):
    """
    Connect to Azure Blob Storage and create the container if necessary.
    """

    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )

    try:
        blob_service_client.create_container(container_name)
        print(f"Container '{container_name}' created.")
    except ResourceExistsError:
        print(f"Container '{container_name}' already exists.")

    return blob_service_client.get_container_client(container_name)


# Check Whether Blob Exists
def blob_exists(container_client, blob_name: str) -> bool:
    """
    Returns True if the blob already exists.
    """
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.exists()


# Upload All PDF Files
def upload_pdf_directory(directory_path: str, container_client):
    """
    Upload all PDF files from a directory.
    Existing files are skipped.
    """

    directory = Path(directory_path)
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return

    pdf_files = list(directory.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"\nFound {len(pdf_files)} PDF files.\n")

    uploaded = 0
    skipped = 0

    for pdf_file in pdf_files:
        blob_name = pdf_file.name

        if blob_exists(container_client, blob_name):
            print(f"Skipped : {blob_name} (already exists)")
            skipped += 1
            continue

        with open(pdf_file, "rb") as data:
            container_client.upload_blob(
                name=blob_name,
                data=data
            )

        print(f"Uploaded: {blob_name}")
        uploaded += 1

    print("\n------------------------------")
    print(f"Uploaded : {uploaded}")
    print(f"Skipped  : {skipped}")
    print("------------------------------")


# List Blob Contents
def list_blobs(container_client):
    """
    List all files inside the blob container.
    """

    print("\nContents of Blob Container")
    print("-" * 40)

    count = 0
    for blob in container_client.list_blobs():
        count += 1
        print(f"{count}. {blob.name} ({blob.size} bytes)")

    if count == 0:
        print("Container is empty.")


# -------------------------------------------------------------------
# Main
if __name__ == "__main__":

    container_client = get_container_client(BLOB_CONTAINER_NAME)
    upload_pdf_directory(
        directory_path=LOCAL_PDF_DIRECTORY,
        container_client=container_client
    )

    print()

    list_blobs(container_client)