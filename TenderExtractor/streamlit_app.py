"""
Streamlit front-end for TenderExtractor.

    streamlit run streamlit_app.py

Upload a PDF -> it's written into config.DATA_UPLOAD_DIR.
Extract -> runs the same run_pipeline() used by main.py on that file.
"""
import threading
import time

import streamlit as st

from app import config
from app.pipeline.runner import run_pipeline
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="TenderExtractor", page_icon="📄")
st.title("TenderExtractor")

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

uploaded_file = st.file_uploader("Choose a tender PDF", type=["pdf"])

col1, col2 = st.columns(2)

with col1:
    if st.button("Upload", disabled=uploaded_file is None, use_container_width=True):
        config.DATA_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = config.DATA_UPLOAD_DIR / uploaded_file.name
        dest_path.write_bytes(uploaded_file.getvalue())
        st.session_state.uploaded_file_name = uploaded_file.name
        st.success(f"Uploaded '{uploaded_file.name}' to {config.DATA_UPLOAD_DIR}")

with col2:
    extract_disabled = st.session_state.uploaded_file_name is None
    if st.button("Extract", disabled=extract_disabled, use_container_width=True):
        blob_name = st.session_state.uploaded_file_name
        result: dict = {}

        def _run():
            try:
                run_pipeline(blob_name)
                result["status"] = "success"
            except Exception as e:
                logger.exception("Pipeline failed for %s", blob_name)
                result["status"] = "error"
                result["error"] = str(e)

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        status_box = st.empty()
        start_time = time.time()
        progress_messages = [
            "Reading PDF pages...",
            "Running OCR...",
            "Extracting NIT details...",
            "Extracting terms & conditions...",
            "Extracting scanned document requirements...",
            "Building Excel files...",
            "Consolidating results...",
        ]
        i = 0
        while worker.is_alive():
            elapsed = int(time.time() - start_time)
            status_box.info(
                f"⏳ {progress_messages[i % len(progress_messages)]} "
                f"({elapsed}s elapsed) — this can take 2-3 minutes, please wait."
            )
            i += 1
            time.sleep(4)

        worker.join()
        status_box.empty()

        if result.get("status") == "success":
            st.success(f"Extraction complete for '{blob_name}'.")
            if config.NOTIFY_ON_COMPLETE:
                st.info(f"📧 An email notification has been sent to {config.NOTIFY_RECIPIENTS}.")
        else:
            st.error(f"Extraction failed for '{blob_name}': {result.get('error')}")