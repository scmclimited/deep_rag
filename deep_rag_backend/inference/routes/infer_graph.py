"""
Infer graph route - Combined ingestion and query using LangGraph pipeline.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from inference.graph.graph_wrapper import ask_with_graph
from ingestion.ingest import ingest as ingest_pdf
from ingestion.ingest_text import ingest_text_file
from ingestion.ingest_image import ingest_image
from retrieval.retrieval import wait_for_chunks
from retrieval.db_utils import get_document_title
from retrieval.thread_tracking.log import log_thread_interaction

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/infer-graph")
async def infer_graph(
    question: str = Form(...),
    attachment: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    thread_id: Optional[str] = Form("default"),
    user_id: Optional[str] = Form(None),  # User ID for thread tracking
    cross_doc: bool = Form(False),
    selected_doc_ids: Optional[str] = Form(None)  # JSON string list of selected doc IDs
):
    """
    Combined ingestion and query endpoint using LangGraph pipeline.
    
    If attachment is provided:
    - Ingest the attachment (PDF, TXT, PNG, JPEG)
    - Then query using the question with LangGraph
    
    If no attachment:
    - Just query existing documents with LangGraph (same as /ask-graph)
    
    Supported file types:
    - PDF: Full text extraction with OCR fallback
    - TXT: Direct text ingestion
    - PNG/JPEG: Image captioning (extracts text via OCR/vision)
    """
    file_ext = None
    content_type = None
    doc_id = None  # Initialize doc_id outside the attachment block
    
    try:
        # Process attachment if provided
        if attachment:
            file_ext = Path(attachment.filename).suffix.lower() if attachment.filename else ""
            content_type = attachment.content_type or ""
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                content = await attachment.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                # Process based on file type
                if file_ext in ['.pdf'] or 'pdf' in content_type:
                    doc_id = ingest_pdf(tmp_path, title=title)
                    logger.info(f"✅ Ingested PDF: {attachment.filename}")
                    logger.info(f"📋 Document ID: {doc_id}")
                    
                elif file_ext in ['.txt'] or 'text/plain' in content_type:
                    doc_id = ingest_text_file(tmp_path, title=title or attachment.filename)
                    logger.info(f"✅ Ingested text file: {attachment.filename}")
                    logger.info(f"📋 Document ID: {doc_id}")
                    
                elif file_ext in ['.png', '.jpg', '.jpeg'] or 'image' in content_type:
                    doc_id = ingest_image(tmp_path, title=title or attachment.filename)
                    logger.info(f"✅ Ingested image: {attachment.filename}")
                    logger.info(f"📋 Document ID: {doc_id}")
                    
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {file_ext}. Supported: PDF, TXT, PNG, JPEG"
                    )
                
                # Wait for chunks to be available before querying
                if doc_id:
                    logger.info(f"Waiting for chunks for document {doc_id}...")
                    try:
                        chunk_count = wait_for_chunks(doc_id, max_wait_seconds=40)
                        logger.info(f"Found {chunk_count} chunks, ready to query")
                    except TimeoutError as e:
                        logger.warning(f"Timeout waiting for chunks: {e}. Proceeding anyway.")
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        # Parse selected_doc_ids if provided (JSON string)
        selected_doc_ids_list = None
        if selected_doc_ids:
            import json
            try:
                selected_doc_ids_list = json.loads(selected_doc_ids)
                if not isinstance(selected_doc_ids_list, list):
                    selected_doc_ids_list = None
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Invalid selected_doc_ids format: {selected_doc_ids}")
                selected_doc_ids_list = None
        
        # Determine which doc_ids to use for retrieval
        # Priority: selected_doc_ids (explicit user selection) > doc_id from upload (if attachment)
        # If both are provided, combine them (user may have selected other docs in addition to uploaded doc)
        doc_ids_to_use = None
        if selected_doc_ids_list is not None:
            # User explicitly provided selected_doc_ids (could be empty list)
            if len(selected_doc_ids_list) > 0:
                doc_ids_to_use = list(selected_doc_ids_list)  # Make a copy to avoid modifying original
                
                # If doc_id from upload is also provided and not already in selected_doc_ids, add it
                # This handles the case where user uploaded a doc AND selected other docs
                if doc_id and doc_id not in doc_ids_to_use:
                    doc_ids_to_use.append(doc_id)
                    logger.info(f"Combining selected_doc_ids with uploaded doc_id: {len(doc_ids_to_use)} document(s) total")
                else:
                    logger.info(f"Using selected_doc_ids: {len(doc_ids_to_use)} document(s)")
            else:
                # Empty list means user explicitly deselected all documents
                # Don't use doc_id from upload if user explicitly deselected
                doc_ids_to_use = None
                logger.info("selected_doc_ids is empty - not filtering to any documents (user deselected all)")
        elif doc_id and attachment:
            # If uploading a document and no selected_doc_ids provided, use that doc_id for the first query
            # (user hasn't had a chance to select it yet)
            doc_ids_to_use = [doc_id]
            logger.info(f"Using doc_id from upload: {doc_id}...")
        elif doc_id:
            # If doc_id provided but no attachment, use it (backward compatibility)
            doc_ids_to_use = [doc_id]
            logger.info(f"Using doc_id: {doc_id}...")
        
        # Run the query with LangGraph
        if cross_doc:
            logger.info("Cross-document retrieval enabled")
        result = ask_with_graph(
            question, 
            thread_id=thread_id, 
            doc_id=doc_id,  # Keep for backward compatibility
            selected_doc_ids=doc_ids_to_use,  # Use selected_doc_ids if provided
            cross_doc=cross_doc
        )
        
        # Get document title if doc_id is available
        final_doc_id = result.get("doc_id") or doc_id
        doc_title = None
        if final_doc_id:
            doc_title = get_document_title(final_doc_id)
        
        # Get doc_ids and pages from result
        doc_ids = result.get("doc_ids", [])
        pages = result.get("pages", [])
        
        # If no doc_id but doc_ids available, use first one
        if not final_doc_id and doc_ids:
            final_doc_id = doc_ids[0]
            doc_title = get_document_title(final_doc_id) if final_doc_id else None
        
        # Log thread interaction to database (synchronous operation, but FastAPI handles it)
        try:
            user_id_for_logging = user_id or "default_user"
            logger.info(f"infer_graph: Logging thread interaction with user_id='{user_id_for_logging}' (from Form user_id='{user_id}')")
            ingestion_meta = None
            if attachment:
                ingestion_meta = {
                    "filename": attachment.filename,
                    "file_type": file_ext or content_type,
                    "doc_id": final_doc_id,
                    "title": title
                }
            record_id = log_thread_interaction(
                user_id=user_id_for_logging,
                thread_id=thread_id,
                query_text=question,
                doc_ids=doc_ids or ([final_doc_id] if final_doc_id else []),
                final_answer=result.get("answer", ""),
                graphstate=result,
                ingestion_meta=ingestion_meta,
                entry_point="rest",
                pipeline_type="langgraph",
                cross_doc=cross_doc
            )
            logger.info(f"infer_graph: Successfully logged thread interaction for user_id='{user_id_for_logging}', thread_id='{thread_id}', record_id={record_id}")
        except Exception as e:
            logger.error(f"Failed to log thread interaction: {e}", exc_info=True)
            # Don't fail the request if logging fails, but log as error
        
        return {
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "action": result.get("action", "answer"),
            "mode": "ingest_and_query" if attachment else "query_only",
            "pipeline": "langgraph",
            "thread_id": thread_id,
            "attachment_processed": attachment is not None,
            "filename": attachment.filename if attachment else None,
            "file_type": file_ext or content_type if attachment else None,
            "doc_id": final_doc_id,
            "doc_ids": doc_ids,  # All doc_ids used
            "doc_title": doc_title,
            "pages": pages,  # Page references
            "cross_doc": cross_doc
        }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /infer-graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

