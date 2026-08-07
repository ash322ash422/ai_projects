# TenderExtractor (POC)

Extracts structured fields from government tender PDFs and outputs a
downloadable Excel summary, using Azure Document Intelligence for
OCR/layout and Azure OpenAI for field extraction.

## Architecture

```
Streamlit UI --> FastAPI Backend --> Azure Blob Storage
                                  --> Azure Document Intelligence (OCR)
                                  --> Azure OpenAI (structured extraction)
                                  --> Excel Generator (openpyxl)
                                  --> Azure Blob Storage --> Download
```

Everything runs synchronously in a single FastAPI process for the POC.
Storage and OCR have local fallbacks (disk + pypdf) so the demo works
even without Azure credentials configured; Azure OpenAI is required
since there is no meaningful local substitute for the LLM extraction step.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set at minimum:
#   AZURE_OPENAI_ENDPOINT
#   AZURE_OPENAI_KEY
#   AZURE_OPENAI_DEPLOYMENT
```

## Run

Terminal 1 - backend:
```bash
uvicorn app.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

Terminal 2 - frontend:
```bash
streamlit run frontend/streamlit_app.py
```
UI: http://localhost:8501

## API

| Method | Path              | Purpose                                  |
| ------ | ----------------- | ----------------------------------------- |
| POST   | /upload            | Upload a tender PDF, returns `job_id`     |
| GET    | /status/{job_id}   | Poll extraction status + extracted fields |
| GET    | /download/{job_id} | Download the generated Excel file         |
| GET    | /health            | Health check                              |

## Tests

```bash
pytest tests/
```

## Docker

```bash
docker build -t tender-extractor .
docker run -p 8000:8000 --env-file .env tender-extractor
```

## Extending to production

See the design doc for the async queue/worker + database architecture
this POC is meant to grow into (Azure Queue/Service Bus, Azure SQL or
Cosmos DB, Azure Entra ID auth, Application Insights). Because each
service in `app/services/` has a single responsibility and the
pipeline is orchestrated from one place (`app/pipeline/extract_pipeline.py`),
moving to that architecture mainly means:

1. Replacing the in-memory `app/utils/job_store.py` with a database table.
2. Calling `extract_pipeline.run()` from a queue-triggered worker instead
   of a FastAPI `BackgroundTask`.
3. Adding auth middleware and structured logging/monitoring.

The prompt template, extraction schema, and Excel layout can all be
reused unchanged for new document types (contracts, invoices, POs) by
swapping `app/prompts/tender_prompt.txt` and `TenderData`.
