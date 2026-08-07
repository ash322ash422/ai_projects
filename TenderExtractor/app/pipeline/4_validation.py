import json
import re
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


BLOB_NAME = "01_tender_mini_version.pdf"
BASE_NAME = Path(BLOB_NAME).stem

# Configuration
# CACHE_EXTRACTED_KEY_DATA = (
#     PROJECT_ROOT / "cache" / "extracted_tender.json"
# )

CACHE_EXTRACTED_KEY_DATA = PROJECT_ROOT / "cache" / f"{BASE_NAME}_EXTRACTED.json"
INPUT_JSON = CACHE_EXTRACTED_KEY_DATA

CACHE_EXTRACTED_KEY_DATA_VALIDATED = (
    PROJECT_ROOT / "cache" / f"{BASE_NAME}_VALIDATED.json"
)
OUTPUT_JSON = CACHE_EXTRACTED_KEY_DATA_VALIDATED


# Validators
def validate_amount(value: str):
    """
    Validates and normalizes an amount.

    Returns
    -------
    {
        "original": "...",
        "normalized": 529995.0,
        "valid": True
    }
    """

    result = {
        "original": value,
        "normalized": None,
        "valid": False
    }

    if not value:
        return result

    cleaned = value

    cleaned = cleaned.replace("Rs.", "")
    cleaned = cleaned.replace("Rs", "")
    cleaned = cleaned.replace("₹", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("/-", "")
    cleaned = cleaned.strip()

    try:
        result["normalized"] = float(cleaned)
        result["valid"] = True

    except Exception:
        pass

    return result


###############################################################################

def validate_date(value: str):
    """
    Validates a date or date-time.

    Supported examples

    17.07.2026

    17:00 hrs. on 17.07.2026
    """

    result = {
        "original": value,
        "normalized": None,
        "valid": False
    }

    if not value:
        return result

    match = re.search(
        r"\d{2}\.\d{2}\.\d{4}",
        value
    )

    if not match:
        return result

    try:

        dt = datetime.strptime(
            match.group(),
            "%d.%m.%Y"
        )

        result["normalized"] = dt.strftime("%Y-%m-%d")
        result["valid"] = True

    except Exception:
        pass

    return result


###############################################################################
# Validation Pipeline
###############################################################################

DATE_FIELDS = [
    "submission_deadline",
    "bid_opening_date"
]

AMOUNT_FIELDS = [
    "estimated_cost",
    "earnest_money"
]


def validate_extracted_json(data):
    """
    Validate extracted fields.

    Original values are preserved.
    """

    validated = {}

    for field, value in data.items():

        if field in DATE_FIELDS:

            validated[field] = validate_date(value)

        elif field in AMOUNT_FIELDS:

            validated[field] = validate_amount(value)

        else:

            validated[field] = {
                "original": value
            }

    return validated


# Main

if __name__ == "__main__":
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        extracted = json.load(f)

    validated = validate_extracted_json(extracted)

    with open( OUTPUT_JSON,"w", encoding="utf-8" ) as f:
        json.dump(
            validated,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("Validation complete.")
    print(f"Saved to {OUTPUT_JSON}")

