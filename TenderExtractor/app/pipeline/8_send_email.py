#!/usr/bin/env python3 8_send_email.py

import sys
from pathlib import Path

# Add app directory to path FIRST
sys.path.append(str(Path(__file__).resolve().parent.parent))

from email_service import send_email_through_azure , build_success_email
from metadata import load_metadata
from config import PROCESSING_METADATA

metadata = load_metadata(PROCESSING_METADATA)

# print(metadata["download_url"])
subject, body = build_success_email(metadata["download_url"])

send_email_through_azure(
    recipient="info@comfortsolutionsgroup.in",
    subject=subject,
    body=body
)
