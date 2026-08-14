"""
Unit tests for tender_extraction_service's page-chunking and cross-batch
merge logic. _extract_chunk is monkeypatched everywhere here, so these
never call the real LLM - they test the pure logic: how pages get split
into batches, and how per-batch results get stitched back together
(re-merging sections whose heading repeats across batches, per
_merge_extractions' docstring).
"""
from app.services import tender_extraction_service as svc
from app.services.tender_extraction_service import (
    ExtractedField,
    ExtractedItem,
    ExtractedSection,
    TenderExtraction,
    _merge_extractions,
    _normalize_heading,
)


def _extraction(**fields) -> TenderExtraction:
    """Build a TenderExtraction with the three required fields, defaulting
    any not passed in to present=False/no sections."""
    defaults = {
        "terms_and_conditions": ExtractedField(present=False),
        "acceptable_make": ExtractedField(present=False),
        "documents_to_be_scanned_and_uploaded": ExtractedField(present=False),
    }
    defaults.update(fields)
    return TenderExtraction(**defaults)


def test_normalize_heading_collapses_whitespace_and_case():
    assert _normalize_heading("  General   Terms &  Conditions ") == "general terms & conditions"
    assert _normalize_heading("General Terms & Conditions") == "general terms & conditions"


def test_normalize_heading_returns_none_for_none():
    assert _normalize_heading(None) is None


def test_merge_extractions_combines_matching_headings_across_chunks():
    chunk1 = _extraction(
        terms_and_conditions=ExtractedField(
            present=True,
            sections=[
                ExtractedSection(
                    heading="General Terms & Conditions",
                    source_pages=[6],
                    items=[ExtractedItem(number="1", text="First clause.")],
                )
            ],
        )
    )
    chunk2 = _extraction(
        terms_and_conditions=ExtractedField(
            present=True,
            sections=[
                ExtractedSection(
                    heading="  General   Terms &  Conditions ",  # same heading, different whitespace/case
                    source_pages=[7],
                    items=[ExtractedItem(number="2", text="Second clause.")],
                )
            ],
        )
    )

    merged = _merge_extractions([chunk1, chunk2])

    assert merged.terms_and_conditions.present is True
    assert len(merged.terms_and_conditions.sections) == 1
    section = merged.terms_and_conditions.sections[0]
    assert section.heading == "General Terms & Conditions"  # first-seen heading text wins
    assert section.source_pages == [6, 7]
    assert [item.text for item in section.items] == ["First clause.", "Second clause."]


def test_merge_extractions_keeps_distinct_headings_as_separate_sections():
    chunk1 = _extraction(
        terms_and_conditions=ExtractedField(
            present=True,
            sections=[ExtractedSection(heading="Terms & Conditions", source_pages=[6])],
        )
    )
    chunk2 = _extraction(
        terms_and_conditions=ExtractedField(
            present=True,
            sections=[ExtractedSection(heading="General Conditions", source_pages=[9])],
        )
    )

    merged = _merge_extractions([chunk1, chunk2])

    headings = {s.heading for s in merged.terms_and_conditions.sections}
    assert headings == {"Terms & Conditions", "General Conditions"}


def test_merge_extractions_never_merges_unheaded_sections_together():
    chunk1 = _extraction(
        acceptable_make=ExtractedField(
            present=True,
            sections=[ExtractedSection(heading=None, source_pages=[3], items=[ExtractedItem(text="MS Pipe: Jindal")])],
        )
    )
    chunk2 = _extraction(
        acceptable_make=ExtractedField(
            present=True,
            sections=[ExtractedSection(heading=None, source_pages=[8], items=[ExtractedItem(text="Cable: Havells")])],
        )
    )

    merged = _merge_extractions([chunk1, chunk2])

    assert len(merged.acceptable_make.sections) == 2
    assert all(s.heading is None for s in merged.acceptable_make.sections)


def test_merge_extractions_present_true_if_any_chunk_found_it():
    chunk1 = _extraction(acceptable_make=ExtractedField(present=False, sections=[]))
    chunk2 = _extraction(
        acceptable_make=ExtractedField(present=True, sections=[ExtractedSection(heading="Approved Makes")])
    )

    merged = _merge_extractions([chunk1, chunk2])

    assert merged.acceptable_make.present is True


def test_extract_data_splits_pages_into_batches_of_requested_size(monkeypatch):
    calls = []

    def fake_extract_chunk(batch, max_pages, token_callback):
        calls.append([p["page_number"] for p in batch])
        if token_callback:
            token_callback(100)
        return _extraction()

    monkeypatch.setattr(svc, "_extract_chunk", fake_extract_chunk)

    pages = [{"page_number": n, "text": f"page {n}"} for n in range(1, 6)]  # 5 pages
    tokens_seen = []

    result = svc.extract_data(pages, page_chunk_size=2, token_callback=tokens_seen.append)

    assert calls == [[1, 2], [3, 4], [5]]  # 3 batches of up to 2 pages each
    assert tokens_seen == [100, 100, 100]
    assert result["terms_and_conditions"]["present"] is False


def test_extract_data_without_page_chunk_size_sends_everything_in_one_call(monkeypatch):
    calls = []

    def fake_extract_chunk(batch, max_pages, token_callback):
        calls.append([p["page_number"] for p in batch])
        return _extraction()

    monkeypatch.setattr(svc, "_extract_chunk", fake_extract_chunk)

    pages = [{"page_number": n, "text": f"page {n}"} for n in range(1, 6)]
    svc.extract_data(pages)

    assert calls == [[1, 2, 3, 4, 5]]
