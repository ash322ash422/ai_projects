"""
Tracks the status of every tender that has gone through the pipeline.

Keyed by blob_name (the PDF's filename) so a batch run can process many
tenders side by side without one job's status overwriting another's -
the old version kept a single processing_metadata.json for the whole
app, which only ever remembered the most recently processed file.
"""
import json
from typing import Optional

from app import config

_LOCK_SAFE_DEFAULT: dict = {}


def _load() -> dict:
    if not config.JOBS_FILE.exists():
        return {}

    with open(config.JOBS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save(jobs: dict) -> None:
    config.JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(config.JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)


def update_job(blob_name: str, **fields) -> dict:
    """
    Merge `fields` into the job record for `blob_name`, creating it if
    this is the first time we've seen this file.
    """
    jobs = _load()
    job = jobs.get(blob_name, {"blob_name": blob_name})
    job.update(fields)
    jobs[blob_name] = job
    _save(jobs)
    return job


def get_job(blob_name: str) -> Optional[dict]:
    return _load().get(blob_name)


def all_jobs() -> dict:
    return _load()
