"""
POST /upload - accepts a tender PDF, kicks off the extraction pipeline
as a background task, and immediately returns a job_id the frontend can
poll via GET /status/{job_id}.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File

from app.pipeline.extract_pipeline import run as run_pipeline
from app.schemas.response import JobStatus, StatusResponse, UploadResponse
from app.utils import job_store
from app.utils.helper import new_job_id, safe_filename
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["upload"])

MAX_FILE_SIZE_MB = 25


@router.post("/upload", response_model=UploadResponse)
async def upload_tender(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadResponse:
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(pdf_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    job_id = new_job_id()
    filename = safe_filename(file.filename or "tender.pdf")

    job_store.create(
        job_id,
        StatusResponse(job_id=job_id, status=JobStatus.QUEUED, filename=filename),
    )

    background_tasks.add_task(run_pipeline, job_id, filename, pdf_bytes)
    logger.info("[%s] Queued %s (%d bytes)", job_id, filename, len(pdf_bytes))

    return UploadResponse(job_id=job_id, status=JobStatus.QUEUED)


@router.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str) -> StatusResponse:
    status = job_store.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return status
