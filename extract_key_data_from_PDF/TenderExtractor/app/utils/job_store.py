"""
Minimal in-memory job store for the POC.

A real deployment (see the "Future Production Architecture" in the
design doc) replaces this with Azure Queue/Service Bus + a database
(Azure SQL / Cosmos DB). For a synchronous single-instance POC, an
in-memory dict is enough to track upload -> processing -> done/failed.
"""

from threading import Lock
from typing import Dict, Optional

from app.schemas.response import StatusResponse

_lock = Lock()
_jobs: Dict[str, StatusResponse] = {}


def create(job_id: str, status: StatusResponse) -> None:
    with _lock:
        _jobs[job_id] = status


def update(job_id: str, **fields) -> None:
    with _lock:
        if job_id not in _jobs:
            return
        current = _jobs[job_id]
        _jobs[job_id] = current.model_copy(update=fields)


def get(job_id: str) -> Optional[StatusResponse]:
    with _lock:
        return _jobs.get(job_id)
