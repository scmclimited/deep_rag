"""
Ask route - Query existing documents using direct pipeline.
"""
import logging
from fastapi import APIRouter, HTTPException
from inference.routes.models import AskBody
from inference.agents import run_deep_rag

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask")
def ask(body: AskBody):
    """
    Query existing documents in the vector database using direct pipeline.
    Assumes documents have already been ingested.
    Uses inference/agents/pipeline.py (direct pipeline without LangGraph).
    
    If doc_id is provided, retrieval is filtered to that specific document.
    If doc_id is not provided, retrieval searches across all documents.
    """
    try:
        if body.doc_id:
            logger.info(f"Querying with document filter: {body.doc_id}...")
        if body.cross_doc:
            logger.info("Cross-document retrieval enabled")
        answer = run_deep_rag(body.question, doc_id=body.doc_id, cross_doc=body.cross_doc)
        return {"answer": answer, "mode": "query_only", "pipeline": "direct", "doc_id": body.doc_id, "cross_doc": body.cross_doc}
    except Exception as e:
        logger.error(f"Error in /ask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

