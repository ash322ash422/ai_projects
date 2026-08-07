"""
API request/response schemas (kept separate from the domain model in
app/models/tender_model.py).
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel

from app.models.tender_model import TenderData


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str = "File received. Processing started."


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    filename: Optional[str] = None
    error: Optional[str] = None
    extracted_data: Optional[TenderData] = None
    download_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    env: str
