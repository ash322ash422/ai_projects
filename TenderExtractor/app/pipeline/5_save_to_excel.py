import json
from pathlib import Path
import sys

import pandas as pd

# Configuration
# PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add the 'app' directory to the system path so 'config' can be found
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import  PROJECT_ROOT
                                 
PDF_NAME = "01_tender_mini_version.pdf"
BASE_NAME = Path(PDF_NAME).stem


INPUT_JSON = (
    PROJECT_ROOT / "cache" / f"{BASE_NAME}_VALIDATED.json"
)

OUTPUT_EXCEL_VALIDATED = (
    PROJECT_ROOT / "cache" / f"{BASE_NAME}_AUDIT.xlsx"
)

OUTPUT_EXCEL_CLEAN = (
    PROJECT_ROOT / "cache" / f"{BASE_NAME}_CLEAN.xlsx"
)


###############################################################################
# Load JSON
###############################################################################

def load_json(json_file):
    """
    Load validated JSON file.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


###############################################################################
# Audit DataFrame
###############################################################################

def validation_json_to_dataframe(data):
    """
    Creates an audit DataFrame.

    Columns:
        field_original
        field_normalized
        field_valid
    """

    row = {}

    for field, info in data.items():

        row[f"{field}_original"] = info.get("original")
        row[f"{field}_normalized"] = info.get("normalized")
        row[f"{field}_valid"] = info.get("valid")

    return pd.DataFrame([row])


###############################################################################
# Clean DataFrame
###############################################################################

def validation_json_to_clean_dataframe(data):
    """
    Creates the final clean DataFrame.

    Rules:
        - If valid=True -> use normalized value
        - Otherwise -> use original value
    """

    row = {}

    for field, info in data.items():

        if info.get("valid", False):
            value = info.get("normalized")
        else:
            value = info.get("original")

        row[field] = value

    return pd.DataFrame([row])


###############################################################################
# Save Excel
###############################################################################

def save_dataframe(df, excel_file):
    """
    Save DataFrame to Excel.
    """

    df.to_excel(
        excel_file,
        index=False
    )


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":

    validation_json = load_json(INPUT_JSON)

    ###########################################################################
    # Audit Excel
    ###########################################################################

    audit_df = validation_json_to_dataframe(validation_json)

    print("\nAudit DataFrame")
    print(audit_df)

    save_dataframe(
        audit_df,
        OUTPUT_EXCEL_VALIDATED
    )

    print(f"\nAudit Excel saved to:\n{OUTPUT_EXCEL_VALIDATED}")

    ###########################################################################
    # Clean Excel
    ###########################################################################

    clean_df = validation_json_to_clean_dataframe(validation_json)

    print("\nClean DataFrame")
    print(clean_df)

    save_dataframe(
        clean_df,
        OUTPUT_EXCEL_CLEAN
    )

    print(f"\nClean Excel saved to:\n{OUTPUT_EXCEL_CLEAN}")