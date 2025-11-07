"""
Ask graph route - Query existing documents using LangGraph pipeline.
"""
import logging
from fastapi import APIRouter, HTTPException
from inference.routes.models import AskGraphBody
from inference.graph.graph_wrapper import ask_with_graph

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask-graph")
def ask_graph(body: AskGraphBody):
    """
    Query existing documents using LangGraph pipeline with conditional routing.
    
    The graph allows agents to decide if they have sufficient evidence
    or need to iterate over query refinement and refine_retrieve options.
    
    If doc_id is provided, retrieval is filtered to that specific document.
    If doc_id is not provided, retrieval searches across all documents.
    """
    try:
        if body.doc_id:
            logger.info(f"Querying with document filter: {body.doc_id}...")
        if body.cross_doc:
            logger.info("Cross-document retrieval enabled")
        answer = ask_with_graph(body.question, thread_id=body.thread_id, doc_id=body.doc_id, cross_doc=body.cross_doc)
        return {
            "answer": answer,
            "mode": "query_only",
            "pipeline": "langgraph",
            "thread_id": body.thread_id,
            "doc_id": body.doc_id,
            "cross_doc": body.cross_doc
        }
    except Exception as e:
        logger.error(f"Error in /ask-graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

