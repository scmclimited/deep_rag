"""
Diagnostics route - Inspect document chunks and pages.
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from retrieval.diagnostics import inspect_document

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/diagnostics/document")
def diagnostics(doc_title: Optional[str] = None, doc_id: Optional[str] = None):
    """
    Inspect what chunks and pages are stored for a document.
    Useful for debugging ingestion and retrieval issues.
    
    Query params:
        doc_title: Document title to search for (partial match)
        doc_id: Document ID (UUID) - if provided, doc_title is ignored
    """
    try:
        result = inspect_document(doc_title=doc_title, doc_id=doc_id)
        return result
    except Exception as e:
        logger.error(f"Error in diagnostics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

