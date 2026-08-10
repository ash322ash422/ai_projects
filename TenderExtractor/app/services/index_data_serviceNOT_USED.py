
"""
Extracts the tender "INDEX" (table of contents) from the OCR data.
The index only ever appears in the first few pages of these tenders,
so the search is deliberately restricted there.
"""
from __future__ import annotations

import difflib
from typing import Optional

TARGET_HEADING = "INDEX"
MAX_PAGES_TO_SEARCH = 3
MATCH_THRESHOLD = 0.7  # below this, we don't trust the "INDEX" match


def _find_index_page(pages: list[dict]) -> tuple[Optional[dict], Optional[str], float]:
    """
    Scan the first MAX_PAGES_TO_SEARCH pages' paragraphs for a line matching
    TARGET_HEADING and return the best (page, matched_line, score) found.
    Paragraphs are used (not `headings`) because the OCR engine doesn't tag
    "INDEX" with a heading role - it just comes through as a plain paragraph.
    """
    best_page, best_line, best_score = None, None, 0.0

    for page in pages[:MAX_PAGES_TO_SEARCH]:
        for para in page.get("paragraphs", []):
            content = (para.get("content") or "").strip()
            if not content:
                continue
            score = difflib.SequenceMatcher(None, TARGET_HEADING.upper(), content.upper()).ratio()
            if score > best_score:
                best_page, best_line, best_score = page, content, score

    return best_page, best_line, best_score


def _parse_index_table(table: list[list[str]]) -> list[dict]:
    """Convert a raw OCR table (list of row-lists, header first) into row dicts."""
    if not table or len(table) < 2:
        return []

    header, *rows = table
    parsed_rows = []
    for row in rows:
        row = row + [""] * (len(header) - len(row))  # defend against short rows
        parsed_rows.append({
            "sl_no": row[0].strip(),
            "description": row[1].strip(),
            "page_no": row[2].strip() if len(row) > 2 else "",
        })
    return parsed_rows


def extract_data(pages: list[dict]) -> dict:
    """
    Find the INDEX section within the first MAX_PAGES_TO_SEARCH pages and
    return its table as structured data.

    Raises ValueError if no INDEX heading/table is found - callers should
    catch this the same way they do for extract_terms_and_condition.
    """
    page, matched_line, score = _find_index_page(pages)

    if page is None or score < MATCH_THRESHOLD:
        raise ValueError(f"'{TARGET_HEADING}' section not found in first {MAX_PAGES_TO_SEARCH} pages.")

    tables = page.get("tables") or []
    if not tables:
        raise ValueError(f"'{TARGET_HEADING}' heading found on page {page['page_number']} but no table present.")

    index_rows = _parse_index_table(tables[0])
    if not index_rows:
        raise ValueError(f"'{TARGET_HEADING}' table on page {page['page_number']} is empty.")

    return {
        "target_heading": TARGET_HEADING,
        "matched_heading": matched_line,
        "match_score": round(score, 4),
        "page_number": page["page_number"],
        "index": index_rows,
    }
