import io
import pdfplumber
from pathlib import Path
import json
import sys

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

from  blob_storage import download_blob_as_stream

# Configuration
BLOB_NAME = "01_tender_mini_version.pdf"
BASE_NAME = Path(BLOB_NAME).stem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

LOCAL_PDF_FILE_TEST = (
    PROJECT_ROOT / "data_uploads" / BLOB_NAME
)

CACHE_FILE = (
    PROJECT_ROOT / "cache" / f"{BASE_NAME}.json"
)


# Add the 'app' directory to the system path so 'config' can be found
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import ( AZURE_STORAGE_CONNECTION_STRING, 
                    AZURE_DOCINTEL_ENDPOINT, 
                    AZURE_DOCINTEL_KEY,
                    AZURE_DOCINTEL_MODEL,
                    BLOB_CONTAINER_NAME,
                    USE_LOCAL_PDF_FILE,
                    USE_CACHE
                )                   




# Client
def create_document_client():
    """
    Create Azure Document Intelligence client.
    """
    return DocumentAnalysisClient(
        endpoint=AZURE_DOCINTEL_ENDPOINT,
        credential=AzureKeyCredential(AZURE_DOCINTEL_KEY)
    )

def validate_pdf_is_digital(pdf_bytes: bytes, sample_pages: int = 5):
    """Checks if PDF bytes contain digital text.

    Raises ValueError if scanned.
    """
    # Wrap bytes in an in-memory file stream
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        pages_to_check = min(total_pages, sample_pages)

        text_found = False

        # Scan the first few pages for selectable text
        for i in range(pages_to_check):
            page_text = pdf.pages[i].extract_text()
            if page_text and len(page_text.strip()) > 10:
                text_found = True
                break

        if not text_found:
            raise ValueError(
                f"Validation Failed: The PDF file appears to be a scanned image. "
                f"No selectable text found in the first {pages_to_check} pages."
            )

    # print("Validation Passed: The PDF is a digital document.")
    return True

def read_local_pdf(pdf_path: str) -> bytes:  # Used for testing with a local PDF file
    """
    Reads a local PDF file and returns its contents as bytes.
    """

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with open(pdf_file, "rb") as f:
        return f.read()


def analyze_document_from_bytes(
    document_bytes,
    endpoint,
    key,
    model_name
):
    """
    Analyze a PDF already loaded into memory.
    """

    client = create_document_client()
    poller = client.begin_analyze_document(
        model_id=model_name,
        document=document_bytes
    )

    return poller.result()

# ------------------------------------------------------------
# Extract Full Text
def extract_text(result):
    """
    Returns the complete document text.
    """
    pages = []
    for page in result.pages:
        page_text = []

        for line in page.lines:
            page_text.append(line.content)

        pages.append("\n".join(page_text))

    return "\n\n".join(pages)


def extract_pages(result):
    pages = []
    for page in result.pages:
        page_text = "\n".join(
            line.content
            for line in page.lines
        )

        pages.append(
            {
                "page_number": page.page_number,
                "text": page_text,
                "tables": [],
                "key_value_pairs": []
            }
        )

    return pages

# Attach tables to pages
def attach_tables(result, pages):

    for table in result.tables:

        if not table.bounding_regions:
            continue

        page_no = table.bounding_regions[0].page_number

        rows = [
            [""] * table.column_count
            for _ in range(table.row_count)
        ]

        for cell in table.cells:

            rows[cell.row_index][cell.column_index] = cell.content

        pages[page_no - 1]["tables"].append(rows)

    return pages

# Attach key-value pairs
def attach_key_value_pairs(result, pages):

    if not result.key_value_pairs:
        return pages

    for kv in result.key_value_pairs:

        if not kv.key.bounding_regions:
            continue

        page_no = kv.key.bounding_regions[0].page_number

        pages[page_no - 1]["key_value_pairs"].append(
            {
                "key": kv.key.content,
                "value": kv.value.content if kv.value else ""
            }
        )

    return pages

def extract_document_data(result):
    pages = extract_pages(result)
    pages = attach_tables(result, pages)
    pages = attach_key_value_pairs(result, pages)

    return {
        "pages": pages
    }

# Save it so that do not have  to call LLM again and again.
# This is useful for testing and development.
def save_document_data(document_data: dict, output_file: str):
    """
    Save extracted document data to a JSON file.
    """

    output_path = Path(output_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            document_data,
            f,
            indent=4,
            ensure_ascii=False
        )

# Load previously extracted document data.
def load_document_data(input_file: str) -> dict:
    """
    Load previously extracted document data.
    """

    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


# Example
if __name__ == "__main__":

    if USE_LOCAL_PDF_FILE:
        document_bytes = read_local_pdf(str(LOCAL_PDF_FILE_TEST)) # Read the local PDF file for testing              
        with pdfplumber.open(io.BytesIO(document_bytes)) as pdf:
            print(f"Number of pages being sent to Azure: {len(pdf.pages)}")
        
        with pdfplumber.open(io.BytesIO(document_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                print(f"Page {i}: {len(text)} characters")
            
            
    else:
        document_bytes = download_blob_as_stream(
        connection_string=AZURE_STORAGE_CONNECTION_STRING,
        container_name=BLOB_CONTAINER_NAME,
        blob_name=BLOB_NAME
        )

    # This will raise a ValueError exception if the PDF is scanned
    if not validate_pdf_is_digital(document_bytes):
        raise ValueError(
                       f"Validation Failed: Only digital PDFs are supported."
                       f" The PDF file appears to be a scanned image."
                   )
       

    if USE_CACHE and CACHE_FILE.exists():
        print("Loading cached extraction...")
        document_data = load_document_data(CACHE_FILE)

    else:
        result = analyze_document_from_bytes(
            document_bytes=document_bytes,
            endpoint=AZURE_DOCINTEL_ENDPOINT,
            key=AZURE_DOCINTEL_KEY,
            model_name=AZURE_DOCINTEL_MODEL
        )
        print("Pages returned:", len(result.pages))
        print("Documents returned:", len(result.documents))

        document_data = extract_document_data(result)
        save_document_data( document_data, CACHE_FILE)    
        
    
    print("=" * 80)
    for page in document_data["pages"]:

        print(f"\nPAGE {page['page_number']}")
        print("=" * 80)

        print("\nTEXT")
        print("-" * 40)
        print(page["text"])

        print("\nKEY VALUE PAIRS")
        print("-" * 40)

        if page["key_value_pairs"]:
            for kv in page["key_value_pairs"]:
                print(f"{kv['key']} : {kv['value']}")
        else:
            print("None")

        print("\nTABLES")
        print("-" * 40)

        if page["tables"]:

            for i, table in enumerate(page["tables"], start=1):

                print(f"\nTable {i}")

                for row in table:
                    print(row)

        else:
            print("None")

        print("\n")

