import json
from pathlib import Path
import sys

from llm import get_llm
from  prompt import build_extraction_prompt, FIELDS_TO_EXTRACT

# Add the 'app' directory to the system path so 'config' can be found
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PAGES_PER_CHUNK, PROJECT_ROOT

BLOB_NAME = "01_tender_mini_version.pdf"
BASE_NAME = Path(BLOB_NAME).stem

CACHE_FILE = (
    PROJECT_ROOT / "cache" / f"{BASE_NAME}.json"
)

CACHE_EXTRACTED_KEY_DATA = PROJECT_ROOT / "cache" / f"{BASE_NAME}_EXTRACTED.json"

# LLM Call
def extract_chunk( llm, pages, fields):
    """
    Sends one chunk of pages to the LLM.
    """

    prompt = build_extraction_prompt(
        pages=pages,
        fields=fields
    )
    print("\n", "=" * 80, "\n")
    print("PROMPT SENT TO LLM:\n", prompt)
    print("\n", "=" * 80, "\n")
  
    response = llm.invoke(prompt)
                          
    return response.content


# Parse JSON
def parse_llm_response(response):
    """
    Converts JSON string returned by the LLM into Python.
    """

    return json.loads(response)


# Merge Partial Results
def merge_results(results):
    """
    Merge outputs from multiple page chunks.

    First non-empty value wins.
    """

    final = {}

    for partial in results:
        for key, value in partial.items():
            if value and not final.get(key):
                final[key] = value

    return final


# Process Entire Document
def extract_document( llm, document_data, fields, pages_per_chunk=2):
    """
    Extract information from the complete document.

    Parameters
    ----------
    document_data : dict

    {
        "pages":[...]
    }

    Returns
    -------
    dict
    """

    pages = document_data["pages"]

    partial_results = []

    for i in range(0, len(pages), pages_per_chunk):
        page_chunk = pages[i:i + pages_per_chunk]
        print(
            f"Processing pages "
            f"{page_chunk[0]['page_number']} "
            f"to "
            f"{page_chunk[-1]['page_number']}"
        )

        response = extract_chunk(
            llm=llm,
            pages=page_chunk,
            fields=fields
        )

        partial_results.append(parse_llm_response(response))

    return merge_results(partial_results)


if __name__ == "__main__":

    # Load document data
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        document_data = json.load(f)

    # Extract information using the LLM
    llm = get_llm()
    extracted_data = extract_document(
        llm=llm,
        document_data=document_data,
        fields=FIELDS_TO_EXTRACT,
        pages_per_chunk=PAGES_PER_CHUNK
    )

    # Save JSON
    with open(CACHE_EXTRACTED_KEY_DATA, "w", encoding="utf-8") as f:
        json.dump(
            extracted_data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nSaved extracted JSON to:\n{CACHE_EXTRACTED_KEY_DATA}")

    # Display Results
    print("\n")
    print("=" * 80)
    print("FINAL EXTRACTION")
    print("=" * 80)

    print(json.dumps(extracted_data, indent=4, ensure_ascii=False ) )

