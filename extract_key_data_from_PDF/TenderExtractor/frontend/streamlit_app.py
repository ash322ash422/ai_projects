"""
Minimal Streamlit UI for the TenderExtractor POC.

Run with:
    streamlit run frontend/streamlit_app.py

Expects the FastAPI backend to be running (default http://localhost:8000).
Set BACKEND_URL env var to point elsewhere (e.g. when deployed to Azure
App Service).
"""

import os
import time

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Tender Extractor", page_icon="📄", layout="centered")
st.title("📄 Tender Extractor (POC)")
st.caption("Upload a government tender PDF and get a structured Excel summary.")

uploaded_file = st.file_uploader("Upload tender PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Extract Tender Data", type="primary"):
        with st.spinner("Uploading and starting extraction..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
                resp.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"Could not reach backend at {BACKEND_URL}: {exc}")
                st.stop()

            job_id = resp.json()["job_id"]

        status_box = st.empty()
        progress = st.progress(0, text="Processing...")

        final_status = None
        for i in range(120):  # poll for up to ~2 minutes
            status_resp = requests.get(f"{BACKEND_URL}/status/{job_id}", timeout=30)
            status_resp.raise_for_status()
            data = status_resp.json()
            status_box.info(f"Status: **{data['status']}**")

            if data["status"] in ("done", "failed"):
                final_status = data
                progress.progress(100, text="Done")
                break

            progress.progress(min(95, (i + 1) * 3), text="Processing...")
            time.sleep(1)

        if final_status is None:
            st.warning("Still processing - refresh status manually or check backend logs.")
        elif final_status["status"] == "failed":
            st.error(f"Extraction failed: {final_status.get('error')}")
        else:
            st.success("Extraction complete!")

            extracted = final_status.get("extracted_data") or {}
            st.subheader("Extracted Fields")
            st.table(
                {
                    "Field": list(extracted.keys()),
                    "Value": [v if v else "—" for v in extracted.values()],
                }
            )

            download_resp = requests.get(f"{BACKEND_URL}/download/{job_id}", timeout=30)
            if download_resp.ok:
                st.download_button(
                    label="⬇️ Download Excel",
                    data=download_resp.content,
                    file_name=f"tender_{job_id}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
else:
    st.info("Upload a PDF tender document to get started.")
