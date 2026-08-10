"""
Turns validated field data into the two Excel deliverables:
  - an audit workbook (original + normalized + valid flag per field)
  - a clean workbook (one tidy value per field, ready to hand off)
"""
from pathlib import Path

import pandas as pd


def validation_json_to_dataframe(data: dict) -> pd.DataFrame:
    row = {}
    for field, info in data.items():
        row[f"{field}_original"] = info.get("original")
        row[f"{field}_normalized"] = info.get("normalized")
        row[f"{field}_valid"] = info.get("valid")
    return pd.DataFrame([row])


def validation_json_to_clean_dataframe(data: dict) -> pd.DataFrame:
    row = {}
    for field, info in data.items():
        # row[field] = info.get("normalized") if info.get("valid", False) else info.get("original") # Use this line if you want to keep the original value when the normalized value is invalid
        row[field] = info.get("original") # Use this line if you want to keep the original value regardless of validity
        
    return pd.DataFrame([row])


def save_dataframe(df: pd.DataFrame, excel_file: Path) -> None:
    excel_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_file, index=False)
