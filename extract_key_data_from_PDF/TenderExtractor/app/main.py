"""
FastAPI entrypoint for TenderExtractor.

Run locally with:
    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import download, health, upload
from app.settings import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="POC: extracts structured fields from government tender "
    "PDFs using Azure Document Intelligence + Azure OpenAI, and returns "
    "an Excel summary.",
    version="0.1.0",
)

# Wide-open CORS for the POC (Streamlit runs on a different port).
# Tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(download.router)


@app.get("/")
def root():
    return {"message": f"{settings.APP_NAME} API is running. See /docs for the API."}
