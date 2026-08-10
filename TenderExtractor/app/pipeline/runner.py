"""
Orchestrates the stages for one tender document, in order, updating
the job store as it goes so progress survives a crash / restart.

This is the *only* place that knows the order of the pipeline - each
stage stays single-responsibility and is unaware of the others.
"""
from app.pipeline import stages
from app.pipeline.context import PipelineContext
from app.services import job_store
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Ordered list of (stage_name, stage_function). Add/remove/reorder here.
STAGES = [
    # ("ingest", stages.stage_ingest),
    # ("ocr", stages.stage_ocr),
    
    ("extract_index", stages.stage_extract_index_data),
    
    ("extract_nit", stages.stage_extract_nit_data),
    ("validate_nit", stages.stage_validate_nit),
    ("export_nit", stages.stage_export_nit_data),
        
    ("extract_scanned_documents", stages.stage_extract_scanned_documents_data),
    ("export_scanned_documents", stages.stage_export_scanned_documents),
    
    ("extract_terms_conditions", stages.stage_extract_terms_conditions_data),
    ("export_terms_conditions", stages.stage_export_terms_conditions),
    
    ("consolidate_all_excels", stages.stage_consolidate_all_excels),
    
    # ("publish", stages.stage_publish),
    # ("notify", stages.stage_notify)

]


def run_pipeline(blob_name: str) -> PipelineContext:
    """
    Runs every stage for one tender. Raises on the first stage failure
    so a caller processing a batch can decide to skip to the next file.
    """
    ctx = PipelineContext(blob_name=blob_name)
    job_store.update_job(blob_name, status="STARTED", error=None)

    for name, stage_fn in STAGES:
        logger.info("[%s] -> %s", blob_name, name)
        try:
            stage_fn(ctx)
        except Exception as exc:
            ctx.status = "FAILED"
            ctx.error = f"{name}: {exc}"
            job_store.update_job(blob_name, 
                                 status=ctx.status, 
                                 error=ctx.error, 
                                 failed_stage=name
            )
            logger.exception("[%s] stage '%s' failed", blob_name, name)
            raise

        job_store.update_job(blob_name, status=f"{name}_done")

    ctx.status = "COMPLETED"
    job_store.update_job(blob_name, status=ctx.status, download_url=ctx.download_url, error=None)
    logger.info("[%s] pipeline completed -> %s", blob_name, ctx.download_url)

    return ctx
