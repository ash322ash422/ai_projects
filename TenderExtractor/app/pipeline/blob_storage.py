
from azure.storage.blob import BlobServiceClient

def blob_service_client(connection_string: str):
    return BlobServiceClient.from_connection_string(connection_string)


def download_blob_as_stream(
    connection_string: str,
    container_name: str,
    blob_name: str
):
    """
    Downloads a blob and returns its bytes.
    """

    blob_service = blob_service_client(connection_string)

    blob_client = blob_service.get_blob_client(
        container=container_name,
        blob=blob_name
    )

    stream = blob_client.download_blob()

    return stream.readall()