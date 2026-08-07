import json
from pathlib import Path

# Save Metadata
def save_metadata(metadata_file: Path, data: dict):
    """
    Save metadata to a JSON file.
    """

    metadata_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(metadata_file, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


###############################################################################
# Load Metadata
def load_metadata(metadata_file: Path):
    """
    Load metadata from a JSON file.
    """

    with open(metadata_file, "r", encoding="utf-8") as f:
        return json.load(f)


###############################################################################
# Update Metadata
def update_metadata(
    metadata_file: Path,
    **kwargs
):
    """
    Update only selected metadata fields.
    """

    try:
        metadata = load_metadata(metadata_file)

    except FileNotFoundError:
        metadata = {}

    metadata.update(kwargs)

    save_metadata(
        metadata_file,
        metadata
    )

    return metadata