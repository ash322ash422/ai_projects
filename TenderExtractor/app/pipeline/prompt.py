import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CACHE_FILE = (
    PROJECT_ROOT / "cache" / "01_tender_mini_version.json"
)

# Fields to Extract
FIELDS_TO_EXTRACT = [
    {
        "name": "nit_number",
        "description": "This is the NIT number."
    },
    {
        "name": "name_of_work_location",
        "description": "Name / Type of the work and location."
    },
    {
        "name": "estimated_cost",
        "description": "Estimated cost put to the tender."
    },
    {
        "name": "earnest_money",
        "description": "Earnest Money Deposit or EMD amount."
    },
    {
        "name": "completion_period",
        "description": "Period of Completion of the work."
    },
    {
        "name": "submission_deadline",
        "description": "Last date and time for bid submission."
    },
    {
        "name": "bid_opening_date",
        "description": "Date and time of bid opening."
    }
]


# JSON Schema Builder
def build_field_schema(fields):
    """
    Creates an empty JSON schema expected from the LLM.
    """

    schema = {}

    for field in fields:
        schema[field["name"]] = ""

    return json.dumps(schema, indent=4)


# Format One Page
def format_page_for_llm(page):
    """
    Converts one extracted page into a structured format
    for the LLM.
    """

    return f"""
        ====================================================
        PAGE {page["page_number"]}
        ====================================================

        TEXT
        -----

        {page["text"]}


        KEY VALUE PAIRS
        ---------------

        {json.dumps(page["key_value_pairs"], indent=4, ensure_ascii=False)}


        TABLES
        ------

        {json.dumps(page["tables"], indent=4, ensure_ascii=False)}

    """


# Format Multiple Pages
def format_pages_for_llm(pages):
    """
    Converts multiple pages into one string.
    """

    return "\n".join(
        format_page_for_llm(page)
        for page in pages
    )


# Prompt Builder
def build_extraction_prompt(
    pages,
    fields
):
    """
    Builds the extraction prompt.
    """

    field_descriptions = "\n".join(
        f"- {field['name']}: {field['description']}"
        for field in fields
    )

    schema = build_field_schema(fields)

    document = format_pages_for_llm(pages)

    prompt = f"""
        You are an expert in analysing Government Tender documents.

        Your task is to extract ONLY the requested fields.

        Use the following priority while searching:

        1. Key Value Pairs
        2. Tables
        3. Text

        Rules

        - Never guess.
        - If a value is missing, return an empty string.
        - Return ONLY valid JSON.
        - Do not explain anything.
        - Do not return Markdown.
        - Do not return code blocks.
        - If the same field appears multiple times, choose the most complete value.

        --------------------------------------------------

        FIELDS TO EXTRACT

        {field_descriptions}

        --------------------------------------------------

        Return EXACTLY this JSON structure

        {schema}

        --------------------------------------------------

        DOCUMENT

        {document}

    """

    return prompt



if __name__ == "__main__":
    # Load document data
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        document_data = json.load(f)

    # Build Prompt
    prompt = build_extraction_prompt(
        pages=document_data["pages"],
        fields=FIELDS_TO_EXTRACT
    )

    # Display Prompt
    print("\n")
    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(prompt)