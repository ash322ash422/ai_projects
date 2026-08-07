"""
GET /download/{job_id} - streams back the generated Excel file once the
pipeline has finished for that job.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.schemas.response import JobStatus
from app.services.blob_service import blob_service
from app.utils import job_store

router = APIRouter(tags=["download"])


@router.get("/download/{job_id}")
def download_excel(job_id: str) -> Response:
    status = job_store.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if status.status != JobStatus.DONE:
        raise HTTPException(status_code=409, detail=f"Job is not ready yet (status={status.status})")

    excel_bytes = blob_service.read_output_file(f"{job_id}.xlsx")
    if excel_bytes is None:
        raise HTTPException(status_code=404, detail="Output file not found")

    filename = f"tender_{job_id}.xlsx"
    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
