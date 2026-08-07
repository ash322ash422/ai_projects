"""
Validates and normalizes the raw JSON returned by the LLM into a
TenderData object, so downstream code (Excel generation) never has to
worry about missing keys or wrong types.
"""

from app.models.tender_model import TenderData
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationService:
    @staticmethod
    def validate(raw_json: dict) -> TenderData:
        cleaned = {}
        for key in TenderData.model_fields:
            value = raw_json.get(key)
            if isinstance(value, (dict, list)):
                value = str(value)
            cleaned[key] = value if value not in ("", "N/A", "n/a") else None

        try:
            return TenderData(**cleaned)
        except Exception as exc:
            logger.error("Validation failed, returning partially filled record: %s", exc)
            return TenderData()


validation_service = ValidationService()
