# NOTE: This did not give satisfactory result
import io
import os
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv

from blob_storage import download_blob_as_stream

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

load_dotenv()

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

LOCAL_PDF_FILE_TEST = (
    Path(__file__).resolve().parent.parent.parent
    / "data_uploads" / "01_tender_mini_version.pdf"
    
)

# ------------------------------------------------------------
# Validation
def validate_pdf_is_digital(pdf_bytes: bytes, sample_pages: int = 5):
    """
    Checks whether the PDF contains selectable text.

    Raises:
        ValueError if the PDF appears to be scanned.
    """

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:

        pages_to_check = min(sample_pages, len(pdf.pages))

        for i in range(pages_to_check):

            text = pdf.pages[i].extract_text()

            if text and len(text.strip()) > 10:
                return True

    raise ValueError(
        "Validation Failed: PDF appears to be scanned. "
        "Only digital PDFs are supported."
    )


def read_local_pdf(pdf_path: str) -> bytes:
    """
    Reads a local PDF file and returns its contents as bytes.
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with open(pdf_file, "rb") as f:
        return f.read()

# ------------------------------------------------------------
# Extract Text
def extract_text(pdf_bytes: bytes):
    """
    Extracts text from every page.

    Returns:
        str
    """

    pages = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:

        for page in pdf.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

    return "\n\n".join(pages)


# ------------------------------------------------------------
# Extract Tables
# ------------------------------------------------------------

def extract_tables(pdf_bytes: bytes):
    """
    Extract tables using pdfplumber.

    Returns:
        list
    """

    tables = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:

        for page in pdf.pages:

            page_tables = page.extract_tables()

            for table in page_tables:
                tables.append(table)

    return tables


# ------------------------------------------------------------
# Extract Per Page
# ------------------------------------------------------------

def extract_pages(pdf_bytes: bytes):
    """
    Returns page-wise text.
    """

    pages = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:

        for index, page in enumerate(pdf.pages):

            pages.append(
                {
                    "page_number": index + 1,
                    "text": page.extract_text() or ""
                }
            )

    return pages


# ------------------------------------------------------------
# Combine Everything
# ------------------------------------------------------------

def extract_document_data(pdf_bytes: bytes):
    """
    Returns everything required by the LLM.
    """

    return {
        "text": extract_text(pdf_bytes),
        "pages": extract_pages(pdf_bytes),
        "tables": extract_tables(pdf_bytes)
    }


# ------------------------------------------------------------
# Example
# ------------------------------------------------------------

if __name__ == "__main__":

    pdf_bytes = read_local_pdf(str(LOCAL_PDF_FILE_TEST)) # Read the local PDF file for testing              
    # pdf_bytes = download_blob_as_stream(
    #     connection_string=CONNECTION_STRING,
    #     container_name="my-automation-container",
    #     blob_name="01_tender_mini_version.pdf"
    # )

    validate_pdf_is_digital(pdf_bytes)

    document_data = extract_document_data(pdf_bytes)

    print("=" * 80)
    print("TEXT")
    print("=" * 80)
    print(document_data["text"])
    print("\n")

    print("=" * 80)
    print("TABLES")
    print("=" * 80)
    print(document_data["tables"])

    print("\n")

    print("=" * 80)
    print("PAGES")
    print("=" * 80)

    for page in document_data["pages"][:2]:
        print(f"Page {page['page_number']}")
        print(page["text"])
        print("-" * 80)