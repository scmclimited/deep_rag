"""
Diagnostics route - Inspect document chunks and pages.
"""
import logging
import json
import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from retrieval.diagnostics import inspect_document

logger = logging.getLogger(__name__)

router = APIRouter()

# Path to diagnostic reports directory
DIAGNOSTIC_REPORTS_DIR = Path(__file__).parent.parent.parent / "retrieval" / "diagnostics" / "diagnostic_reports"


def save_diagnostic_report(result: dict, doc_id: str, file_type: Optional[str] = None):
    """
    Save diagnostic report to JSON file.
    
    Args:
        result: Diagnostic report dictionary
        doc_id: Document ID
        file_type: Optional file type prefix (ocr, png, txt)
    """
    try:
        # Ensure directory exists
        DIAGNOSTIC_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Determine filename based on file type
        if file_type:
            filename = f"{file_type}_doc_id={doc_id}.json"
        else:
            filename = f"doc_id={doc_id}.json"
        
        file_path = DIAGNOSTIC_REPORTS_DIR / filename
        
        # Save JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved diagnostic report to {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving diagnostic report: {e}", exc_info=True)
        raise


def get_file_type_from_doc(result: dict) -> Optional[str]:
    """
    Determine file type from diagnostic result.
    
    Args:
        result: Diagnostic report dictionary
        
    Returns:
        File type prefix (ocr, png, txt) or None
    """
    if "error" in result or "documents" in result:
        return None
    
    # Check source_path for file extension
    doc = result.get("document", {})
    source_path = doc.get("source_path", "")
    
    if not source_path:
        return None
    
    path_lower = source_path.lower()
    if path_lower.endswith('.png') or path_lower.endswith('.jpg') or path_lower.endswith('.jpeg'):
        # Check if it's OCR by looking at chunks
        page_dist = result.get("page_distribution", {})
        for page_info in page_dist.values():
            chunks = page_info.get("chunks", [])
            for chunk in chunks:
                if chunk.get("is_ocr", False):
                    return "ocr"
        return "png"
    elif path_lower.endswith('.txt'):
        return "txt"
    
    return None


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
        
        # Save diagnostic report to JSON file if doc_id is provided
        if doc_id and "error" not in result and "documents" not in result:
            file_type = get_file_type_from_doc(result)
            try:
                save_diagnostic_report(result, doc_id, file_type)
            except Exception as e:
                logger.warning(f"Failed to save diagnostic report: {e}")
        
        return result
    except Exception as e:
        logger.error(f"Error in diagnostics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

