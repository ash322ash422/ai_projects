from dataclasses import dataclass
from pathlib import Path

from config import (
    DATA_UPLOAD_DIR,
    CACHE_DIR,
    OUTPUT_DIR
)


@dataclass
class TenderSettings:

    blob_name: str

    @property
    def pdf_file(self):

        return DATA_UPLOAD_DIR / self.blob_name

    @property
    def cache_document(self):

        return CACHE_DIR / f"{Path(self.blob_name).stem}.json"

    @property
    def extracted_json(self):

        return CACHE_DIR / (
            f"{Path(self.blob_name).stem}_extracted.json"
        )

    @property
    def validated_json(self):

        return CACHE_DIR / (
            f"{Path(self.blob_name).stem}_validated.json"
        )

    @property
    def audit_excel(self):

        return OUTPUT_DIR / (
            f"{Path(self.blob_name).stem}_audit.xlsx"
        )

    @property
    def clean_excel(self):

        return OUTPUT_DIR / (
            f"{Path(self.blob_name).stem}.xlsx"
        )