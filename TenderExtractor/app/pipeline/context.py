# context.py
"""
PipelineContext is the single object threaded through every stage.
Each stage reads what it needs off the context and writes its own
result back onto it - stages never call each other directly, which is
what keeps them independently testable and reorderable.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app import config

@dataclass
class PipelineContext:
    blob_name: str  # e.g. "01_tender_mini_version.pdf"

    # populated as stages run
    pdf_bytes: Optional[bytes] = None
    document_data: Optional[dict] = None    # OCR output: {"pages": [...]}
    
    extracted_index_data: Optional[dict] = None   # raw INDEX table from the LLM/OCR
    
    extracted_nit_data: Optional[dict] = None   # raw field values like NIT, etc. from the LLM
    validated_nit_data: Optional[dict] = None   # normalized + valid-flagged fields
    
    extracted_terms_conditions_data: Optional[dict] = None   # raw field values from the LLM
    
    extracted_scanned_documents_requirements_data: Optional[dict] = None   # raw field values from the LLM
    
    download_url: Optional[str] = None

    status: str = "PENDING"
    error: Optional[str] = None

    @property
    def base_name(self) -> str:
        return Path(self.blob_name).stem

    @property
    def local_pdf_path(self) -> Path:
        return config.DATA_UPLOAD_DIR / self.blob_name

    @property
    def ocr_cache_path(self) -> Path:
        return config.CACHE_DIR / f"{self.base_name}.ocr.json"

    @property
    def extracted_index_cache_path(self) -> Path:
        return config.CACHE_DIR / f"{self.base_name}.index.json"

    @property
    def extracted_nit_cache_path(self) -> Path:
        return config.CACHE_DIR / f"{self.base_name}.extracted_nit.json"

    @property
    def validated_nit_cache_path(self) -> Path:
        return config.CACHE_DIR / f"{self.base_name}.validated_nit.json"

    @property
    def audit_nit_excel_path(self) -> Path:
        return config.OUTPUT_DIR / f"{self.base_name}_nit_audit.xlsx"

    @property
    def clean_nit_excel_path(self) -> Path:
        return config.OUTPUT_DIR / f"{self.base_name}_nit_clean.xlsx"

    @property
    def extracted_terms_conditions_cache_path(self) -> Path:
        return config.CACHE_DIR / f"{self.base_name}.terms_conditions.json"

    @property
    def terms_conditions_excel_path(self) -> Path:
        return config.OUTPUT_DIR / f"{self.base_name}_terms_conditions.xlsx"


    @property
    def extracted_scanned_document_requirements_cache_path(self) -> Path:
        return config.CACHE_DIR / f"{self.base_name}.scanned_document_requirements.json"

    @property
    def scanned_documents_excel_path(self) -> Path:
        return config.OUTPUT_DIR / f"{self.base_name}_scanned_document_requirements.xlsx"
    
    @property
    def consolidated_excel_path(self) -> Path:
        return config.OUTPUT_DIR / f"{self.base_name}.xlsx"
    

