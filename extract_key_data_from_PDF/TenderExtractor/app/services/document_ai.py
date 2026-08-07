"""
Wraps Azure AI Document Intelligence (formerly Form Recognizer) to turn a
tender PDF into plain text: paragraphs + tables laid out in reading order.

If AZURE_DOCINTEL_ENDPOINT / KEY are not configured, falls back to a local
PDF text extraction using pypdf so the pipeline still runs end-to-end for
a demo (with lower quality on scanned/complex PDFs - this is exactly why
Document Intelligence is recommended for the production build).
"""

from io import BytesIO

from app.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    DocumentIntelligenceClient = None
    AzureKeyCredential = None


class DocumentAIService:
    def __init__(self) -> None:
        self.use_azure = bool(
            settings.AZURE_DOCINTEL_ENDPOINT
            and settings.AZURE_DOCINTEL_KEY
            and DocumentIntelligenceClient is not None
        )
        if self.use_azure:
            self.client = DocumentIntelligenceClient(
                endpoint=settings.AZURE_DOCINTEL_ENDPOINT,
                credential=AzureKeyCredential(settings.AZURE_DOCINTEL_KEY),
            )
        else:
            logger.warning(
                "AZURE_DOCINTEL_ENDPOINT/KEY not set - falling back to local "
                "pypdf text extraction (POC only, no OCR/table understanding)."
            )

    def extract_text(self, pdf_bytes: bytes) -> str:
        if self.use_azure:
            return self._extract_with_azure(pdf_bytes)
        return self._extract_with_pypdf(pdf_bytes)

    # ---------------------------------------------------------------
    def _extract_with_azure(self, pdf_bytes: bytes) -> str:
        poller = self.client.begin_analyze_document(
            settings.AZURE_DOCINTEL_MODEL,
            body=pdf_bytes,
            content_type="application/pdf",
        )
        result = poller.result()

        chunks = []
        for paragraph in getattr(result, "paragraphs", []) or []:
            if paragraph.content:
                chunks.append(paragraph.content)

        for table in getattr(result, "tables", []) or []:
            chunks.append(self._table_to_text(table))

        return "\n".join(chunks)

    @staticmethod
    def _table_to_text(table) -> str:
        rows = {}
        for cell in table.cells:
            rows.setdefault(cell.row_index, {})[cell.column_index] = cell.content
        lines = []
        for row_idx in sorted(rows):
            row = rows[row_idx]
            lines.append(" | ".join(row[c] for c in sorted(row)))
        return "\n".join(lines)

    # ---------------------------------------------------------------
    @staticmethod
    def _extract_with_pypdf(pdf_bytes: bytes) -> str:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_bytes))
        text_parts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(text_parts)


document_ai_service = DocumentAIService()
