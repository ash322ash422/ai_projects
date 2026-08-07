"""
Lightweight tests for the parts of the pipeline that don't require live
Azure credentials: JSON extraction, validation, and Excel generation.

Run with:
    pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.tender_model import TenderData
from app.services.excel_service import excel_service
from app.services.validation_service import validation_service
from app.utils.helper import extract_json_block, new_job_id, safe_filename


def test_new_job_id_is_unique():
    assert new_job_id() != new_job_id()


def test_safe_filename_strips_bad_chars():
    # Path(...).name already discards any directory components,
    # then remaining unsafe characters (space, ?) are replaced.
    assert safe_filename("../../evil name?.pdf") == "evil_name_.pdf"


def test_extract_json_block_handles_markdown_fence():
    raw = '```json\n{"tender_number": "GEM/2026/1"}\n```'
    parsed = extract_json_block(raw)
    assert parsed["tender_number"] == "GEM/2026/1"


def test_validation_fills_missing_fields_with_none():
    tender = validation_service.validate({"tender_number": "GEM/2026/1"})
    assert tender.tender_number == "GEM/2026/1"
    assert tender.email is None


def test_excel_service_builds_a_workbook():
    tender = TenderData(tender_number="GEM/2026/1", organization="Ministry of Railways")
    output = excel_service.build(tender, source_filename="sample.pdf")
    assert output[:2] == b"PK"  # xlsx files are zip archives
