"""
Sends OCR'd pages to the LLM in chunks and merges the partial results
into one flat dict of extracted fields.
"""
import json

from app.services.prompt import build_extraction_prompt
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def extract_chunk(llm, pages: list, fields: list) -> dict:
    prompt = build_extraction_prompt(pages=pages, fields=fields)
    logger.info(".....Sending %s pages to LLM for extraction", len(pages))
    
    print("prompt sent to LLM:\n", prompt)
    response = llm.invoke(prompt)
    print("response from LLM:\n", response.content)
    
    return json.loads(response.content)

def merge_results(results: list) -> dict:
    """First non-empty value wins across chunks."""
    final = {}
    for partial in results:
        for key, value in partial.items():
            if value and not final.get(key):
                final[key] = value
    return final


# def extract_document(llm, document_data: dict, fields: list, pages_per_chunk: int = 2) -> dict:
#     pages = document_data["pages"]
#     partial_results = []

#     for i in range(0, len(pages), pages_per_chunk):
#         chunk = pages[i : i + pages_per_chunk]
#         logger.info("Extracting pages %s-%s", chunk[0]["page_number"], chunk[-1]["page_number"])
#         partial_results.append(extract_chunk(llm, chunk, fields))

#     return merge_results(partial_results)


def extract_document(llm, document_data: dict, fields: list, pages_per_chunk: int = 2) -> dict:
    if document_data is None:
        logger.error("Received None for document_data.")
        return {}

    # NIT data is present only in  first 4 pages
    pages = document_data.get("pages", [])[:4]
    
    # 1. Guard Clause for Empty Input
    if not pages:
        logger.warning("No pages found in document_data. Skipping extraction.")
        return {}

    partial_results = []

    # 2. Safe Chunked Iteration
    for i in range(0, len(pages), pages_per_chunk):
        chunk = pages[i : i + pages_per_chunk]
        
        # Explicit check to guarantee chunk elements exist before logging
        if chunk:
            start_page = chunk[0].get("page_number", i + 1)
            end_page = chunk[-1].get("page_number", i + len(chunk))
            logger.info("Extracting pages %s-%s", start_page, end_page)
        
        # 3. Process Chunk
        result = extract_chunk(llm, chunk, fields)
        if result:  # Only append if the LLM successfully returned data
            partial_results.append(result)

    # 4. Defensive Merge
    if not partial_results:
        return {}
        
    return merge_results(partial_results)
