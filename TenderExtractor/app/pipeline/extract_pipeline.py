"""
Orchestrates the end-to-end extraction pipeline:

  1. Store the uploaded PDF (Blob Storage / local fallback)
  2. Run Azure Document Intelligence (OCR + layout + tables)
  3. Run Azure OpenAI to extract structured fields as JSON
  4. Validate the JSON against TenderData
  5. Generate the output Excel file
  6. Store the Excel file and return a download reference

This module is intentionally the only place that knows the *order* of
the services - each service itself stays single-responsibility and
independently testable.
"""

from app.schemas.response import JobStatus, StatusResponse
from app.services.blob_service import blob_service
from app.services.document_ai import document_ai_service
from app.services.excel_service import excel_service
from app.services.openai_service import openai_service
from app.services.validation_service import validation_service
from app.utils import job_store
from app.utils.logger import get_logger

logger = get_logger(__name__)


def run(job_id: str, filename: str, pdf_bytes: bytes) -> None:
    """Runs synchronously and updates job_store as it progresses.

    Called from a FastAPI BackgroundTask so the /upload endpoint can
    return immediately with a job_id for the frontend to poll.
    """
    try:
        job_store.update(job_id, status=JobStatus.PROCESSING)

        # 1. Persist the input file
        blob_service.upload_input_file(filename, pdf_bytes)

        # 2. OCR / layout extraction
        logger.info("[%s] Running Document Intelligence on %s", job_id, filename)
        document_text = document_ai_service.extract_text(pdf_bytes)
        if not document_text.strip():
            raise ValueError("No text could be extracted from the document")

        # 3. LLM structured extraction
        logger.info("[%s] Calling Azure OpenAI for field extraction", job_id)
        raw_json = openai_service.extract_fields(document_text)

        # 4. Validate
        tender_data = validation_service.validate(raw_json)

        # 5. Build Excel
        excel_bytes = excel_service.build(tender_data, source_filename=filename)

        # 6. Store output + finish
        output_filename = f"{job_id}.xlsx"
        blob_service.upload_output_file(output_filename, excel_bytes)

        job_store.update(
            job_id,
            status=JobStatus.DONE,
            extracted_data=tender_data,
            download_url=f"/download/{job_id}",
        )
        logger.info("[%s] Completed successfully", job_id)

    except Exception as exc:
        logger.exception("[%s] Pipeline failed", job_id)
        job_store.update(job_id, status=JobStatus.FAILED, error=str(exc))
