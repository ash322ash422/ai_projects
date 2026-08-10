# TenderExtractor

Turns a government tender PDF into a validated Excel summary:

```
PDF  ->  OCR (Document Intelligence)  ->  LLM field extraction  ->
validate/normalize  ->  Excel  ->  publish  ->  notify
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in Azure OpenAI + Document Intelligence keys

# drop PDFs into data_uploads/, then:
python main.py                # processes every PDF in data_uploads/
python main.py --file 01_tender_mini_version.pdf   # or just one
```

Output lands in `output/<name>.xlsx` (clean) and `output/<name>_audit.xlsx`
(original + normalized value + valid flag, for QA). Progress for every file
is tracked in `cache/jobs.json`, keyed by filename, so a batch run of many
tenders never overwrites one job's status with another's.

## Layout

```
app/
  config.py               # the only module that reads os.environ
  pipeline/
    context.py             # PipelineContext - the object passed between stages
    stages.py               # one function per pipeline step
    runner.py               # runs the stages in order, updates job status
  services/                # one file per external concern, no pipeline knowledge
    blob_storage.py
    document_intelligence.py
    llm.py
    prompt.py
    extraction.py
    validation.py
    excel_export.py
    email_service.py
    job_store.py
  utils/
    logging_config.py
main.py                    # CLI: discovers PDFs and runs the pipeline over them
tests/
```

**Services** know how to talk to one external thing (Blob Storage, Document
Intelligence, the LLM, email) and nothing about tenders or the pipeline.
**Stages** know the tender domain and call one or two services each.
**The runner** knows the *order* of stages; nothing else does.

## Why it's structured this way

- **Modular** - each stage is a plain function `stage_x(ctx) -> None`. Add a
  step by writing one function and adding one line to `STAGES` in
  `runner.py`. Nothing else has to change.
- **Scales to a batch** - `main.py` loops over every PDF in `data_uploads/`
  (or a Blob container, once `USE_LOCAL_PDF_FILE=false`), and one tender
  failing doesn't stop the rest. Each tender's status lives in its own entry
  in `cache/jobs.json`.
- **Cheap to re-run** - OCR and LLM calls are the expensive/slow steps, so
  their output is cached to disk per-file (`USE_CACHE=true`). Re-running the
  pipeline after a code change in a later stage doesn't re-call Azure.
- **Degrades gracefully** - if `AZURE_STORAGE_CONNECTION_STRING` isn't set,
  the `publish` stage just leaves the Excel file in `output/` instead of
  failing. Good for local development; nothing to configure for a POC.
- **Not over-engineered** - no web framework, no async, no class
  hierarchies. It's a list of functions run in order. That's the whole
  abstraction, which is enough for a linear document pipeline like this one.

## What changed from the previous version

- Seven numbered scripts you had to run by hand, each hardcoding the same
  filename, are now one pipeline you can point at a folder.
- Every tender now gets its own job record (`cache/jobs.json`) instead of a
  single `processing_metadata.json` that only remembered the last file
  processed.
- Fixed a couple of latent bugs: `settings.py` referenced a
  `DATA_UPLOAD_DIR` that didn't exist in `config.py`; the Blob container
  names in `.env.example` (`AZURE_STORAGE_CONTAINER_UPLOADS/OUTPUT`) were
  never actually read - `config.py` hardcoded different names instead.
- Removed dead code (`*_NOT_USED.py` files, a notebook, a stray
  `extracted_tables.xlsx`) and a test file that imported modules that don't
  exist anywhere in the project.
- Replaced scattered `print()` calls with a shared logger.
- The recipient email address was hardcoded inside `8_send_email.py`; it's
  now `NOTIFY_RECIPIENT` in `.env`, and the whole notify step is optional
  and non-fatal (a bounced email no longer means a successfully processed
  tender gets marked FAILED).

## Adding a new field to extract

1. Add `{"name": ..., "description": ...}` to `FIELDS_TO_EXTRACT` in
   `app/services/prompt.py`.
2. If it's a date or currency amount, add its name to `DATE_FIELDS` /
   `AMOUNT_FIELDS` in `app/services/validation.py` so it gets normalized
   automatically. Otherwise it's carried through as-is.

## Tests

```bash
pytest tests/
```

Covers prompt building, chunk-merging, and field validation - the parts
that don't need live Azure credentials to test.
