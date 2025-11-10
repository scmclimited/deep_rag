# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUntypedFunctionDecorator=false

"""
Ask graph route - Query existing documents using LangGraph pipeline.
"""
import logging
from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, HTTPException

from inference.routes.models import AskGraphBody
from inference.graph.graph_wrapper import ask_with_graph
from retrieval.db_utils import get_document_title
from retrieval.thread_tracking.log import log_thread_interaction

logger = logging.getLogger(__name__)

router = APIRouter()


# type: ignore[reportUnknownMemberType] for pyright decorator inference
@router.post("/ask-graph")  # pyright: ignore[reportUnknownMemberType]
def ask_graph(body: AskGraphBody) -> Dict[str, Any]:
    """
    Query existing documents using LangGraph pipeline with conditional routing.
    
    The graph allows agents to decide if they have sufficient evidence
    or need to iterate over query refinement and refine_retrieve options.
    
    If doc_id is provided, retrieval is filtered to that specific document.
    If doc_id is not provided, retrieval searches across all documents.
    """
    try:
        # Handle multi-document selection (selected_doc_ids) or single doc_id
        # CRITICAL: If selected_doc_ids is explicitly provided (even if empty), use it
        # This prevents using doc_id from previous queries when user explicitly deselected
        doc_ids_to_use: Optional[List[str]] = None
        explicit_empty_selection = False
        if body.selected_doc_ids is not None:
            # selected_doc_ids was explicitly provided (could be empty list)
            if len(body.selected_doc_ids) > 0:
                doc_ids_to_use = body.selected_doc_ids
                logger.info(f"Querying with multi-document selection: {len(doc_ids_to_use)} document(s)")
            else:
                # Empty list means user explicitly deselected all documents
                explicit_empty_selection = True
                if body.cross_doc:
                    # For cross-doc search, treat explicit deselection as "search all"
                    doc_ids_to_use = None
                    logger.info("selected_doc_ids empty but cross_doc=True - searching across all documents")
                else:
                    doc_ids_to_use = []  # Explicitly empty for non-cross-doc queries
                logger.info("selected_doc_ids is empty - user explicitly deselected all documents")
        elif body.doc_id:
            # Fallback to doc_id only if selected_doc_ids was not provided (None)
            doc_ids_to_use = [body.doc_id]
            logger.info(f"Querying with document filter: {body.doc_id}...")
        
        explicit_empty_selection = explicit_empty_selection or (
            body.selected_doc_ids is not None and len(body.selected_doc_ids) == 0
        )
        if doc_ids_to_use and len(doc_ids_to_use) > 0:
            if len(doc_ids_to_use) > 1:
                logger.info(f"Querying with multi-document selection: {len(doc_ids_to_use)} document(s)")
            else:
                logger.info(f"Querying with document filter: {doc_ids_to_use[0]}...")
        elif explicit_empty_selection and not body.cross_doc:
            logger.info("No documents selected and cross_doc=False - returning empty response")
            return {
                "answer": "No documents selected. Choose a document from the sidebar, attach a document to your next message, or enable Cross-Document Search.",
                "confidence": 0.0,
                "action": "no_documents",
                "mode": "query_only",
                "pipeline": "langgraph",
                "thread_id": body.thread_id,
                "doc_id": None,
                "doc_ids": [],
                "doc_title": None,
                "pages": [],
                "cross_doc": body.cross_doc
            }
        elif not body.cross_doc:
            # No documents selected and cross_doc disabled leads to no evidence; return early.
            logger.info("No documents selected; cross_doc=False. Returning no-documents response.")
            return {
                "answer": "No documents selected. Choose a document from the sidebar or enable Cross-Document Search.",
                "confidence": 0.0,
                "action": "no_documents",
                "mode": "query_only",
                "pipeline": "langgraph",
                "thread_id": body.thread_id,
                "doc_id": None,
                "doc_ids": [],
                "doc_title": None,
                "pages": [],
                "cross_doc": body.cross_doc
            }
        
        if body.cross_doc:
            logger.info("Cross-document retrieval enabled")
        
        # CRITICAL: Don't pass doc_id if selected_doc_ids is explicitly empty
        # This prevents using persisted doc_id from previous queries
        doc_id_to_pass = None if explicit_empty_selection else body.doc_id
        
        thread_id_value: str = body.thread_id or "default"
        
        result = cast(
            Dict[str, Any],
            ask_with_graph(
            body.question, 
            thread_id=thread_id_value,
            doc_id=doc_id_to_pass,  # Only pass doc_id if not explicitly deselected
            selected_doc_ids=doc_ids_to_use,  # Pass empty list if explicitly deselected
            cross_doc=body.cross_doc,
            ),
        )
        
        # Get document title if doc_id is provided
        doc_id_value = result.get("doc_id")
        doc_id: Optional[str] = doc_id_value if isinstance(doc_id_value, str) else body.doc_id
        doc_title = None
        doc_titles_map: Dict[str, Optional[str]] = {}
        if doc_id:
            doc_title = get_document_title(doc_id)
            doc_titles_map[doc_id] = doc_title
        
        # Get doc_ids and pages from result
        doc_ids_raw = result.get("doc_ids", [])
        doc_ids: List[str] = [str(value) for value in doc_ids_raw if value is not None]
        pages_raw = result.get("pages", [])
        pages: List[str] = [str(value) for value in pages_raw if value is not None]
        
        # If no doc_id but doc_ids available, use first one
        if not doc_id and doc_ids:
            doc_id = doc_ids[0]
            doc_title = get_document_title(doc_id) if doc_id else None
            if doc_id:
                doc_titles_map[doc_id] = doc_title
        
        doc_titles: List[Optional[str]] = []
        if len(doc_ids) > 1:
            for doc_identifier in doc_ids:
                if doc_identifier not in doc_titles_map:
                    doc_titles_map[doc_identifier] = get_document_title(doc_identifier)
                doc_titles.append(doc_titles_map.get(doc_identifier))
        
        # Log thread interaction to database (synchronous operation, but FastAPI handles it)
        try:
            user_id = body.user_id or "default_user"
            logger.info(f"ask_graph: Logging thread interaction with user_id='{user_id}' (from body.user_id='{body.user_id}')")
            record_id = log_thread_interaction(
                user_id=user_id,
                thread_id=thread_id_value,
                query_text=body.question,
                doc_ids=doc_ids or ([doc_id] if doc_id else []),
                final_answer=str(result.get("answer", "")),
                graphstate=result,
                entry_point="rest",
                pipeline_type="langgraph",
                cross_doc=body.cross_doc
            )
            logger.info(f"ask_graph: Successfully logged thread interaction for user_id='{user_id}', thread_id='{body.thread_id}', record_id={record_id}")
        except Exception as e:
            logger.error(f"Failed to log thread interaction: {e}", exc_info=True)
            # Don't fail the request if logging fails, but log as error
        
        return {
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "action": result.get("action", "answer"),
            "mode": "query_only",
            "pipeline": "langgraph",
            "thread_id": body.thread_id,
            "doc_id": doc_id,
            "doc_ids": doc_ids,  # All doc_ids used
            "doc_title": doc_title,
            "doc_titles": doc_titles if doc_titles else None,
            "pages": pages,  # Page references
            "cross_doc": body.cross_doc
        }
    except Exception as e:
        logger.error(f"Error in /ask-graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

