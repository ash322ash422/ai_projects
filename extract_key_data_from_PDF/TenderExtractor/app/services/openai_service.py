"""
Wraps Azure OpenAI to turn raw tender text into structured JSON matching
TenderData.

If Azure OpenAI credentials are not configured, raises a clear runtime
error - unlike storage/OCR there's no meaningful local fallback for the
LLM extraction step.
"""

from pathlib import Path

from app.settings import settings
from app.utils.helper import extract_json_block
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "tender_prompt.txt"


class OpenAIService:
    def __init__(self) -> None:
        self.configured = bool(
            settings.AZURE_OPENAI_ENDPOINT
            and settings.AZURE_OPENAI_KEY
            and AzureOpenAI is not None
        )
        if self.configured:
            self.client = AzureOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
            )
        else:
            logger.warning(
                "AZURE_OPENAI_ENDPOINT/KEY not set - the extraction step will "
                "fail until Azure OpenAI is configured in .env."
            )
        self.prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    def extract_fields(self, document_text: str) -> dict:
        if not self.configured:
            raise RuntimeError(
                "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT and "
                "AZURE_OPENAI_KEY in your .env file."
            )

        prompt = self.prompt_template.format(document_text=document_text[:60000])

        response = self.client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You extract structured data and reply with JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=settings.AZURE_OPENAI_MAX_TOKENS,
            temperature=settings.AZURE_OPENAI_TEMPERATURE,
        )

        raw_text = response.choices[0].message.content or ""
        return extract_json_block(raw_text)


openai_service = OpenAIService()
