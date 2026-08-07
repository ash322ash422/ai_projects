"""
Small stateless helper functions used across services.
"""

import json
import re
import uuid
from pathlib import Path


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def safe_filename(filename: str) -> str:
    """Strip path separators and unsafe characters from a user-supplied name."""
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "file.pdf"


def extract_json_block(text: str) -> dict:
    """
    LLMs occasionally wrap JSON in markdown fences or add stray text.
    This pulls out the first {...} block and parses it, raising a clear
    error if nothing valid is found.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")

    return json.loads(match.group(0))
