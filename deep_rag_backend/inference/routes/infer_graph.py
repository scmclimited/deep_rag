"""
Infer graph route - Combined ingestion and query using LangGraph pipeline.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
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
    attachments: Optional[List[UploadFile]] = File(None),
    attachment: Optional[UploadFile] = File(None),  # Backward compatibility for single attachment clients
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
    thread_id_value = thread_id or "default"
    uploaded_doc_ids: List[str] = []
    attachment_metadata: List[Dict[str, Any]] = []
    attachments_list: List[UploadFile] = []
    if attachments:
        attachments_list.extend([item for item in attachments if item is not None])
    if attachment:
        attachments_list.append(attachment)
    # Deduplicate while preserving order (FormData may pass same object twice)
    unique_attachments: List[UploadFile] = []
    seen_ids = set()
    for upload in attachments_list:
        if upload is None:
            continue
        key = id(upload)
        if key in seen_ids:
            continue
        seen_ids.add(key)
        unique_attachments.append(upload)
    attachments_list = unique_attachments
    doc_id = None  # Primary doc_id for backward compatibility
    
    try:
        # Process attachments if provided
        if attachments_list:
            logger.info(f"Ingesting {len(attachments_list)} attachment(s)")
        for idx, upload in enumerate(attachments_list):
            file_ext = Path(upload.filename).suffix.lower() if upload.filename else ""
            content_type = upload.content_type or ""
            tmp_path = None
            generated_doc_id = None

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                content = await upload.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name

            try:
                title_to_use = title if title and len(attachments_list) == 1 else (upload.filename or title or "Uploaded Document")
                if not title_to_use:
                    title_to_use = "Uploaded Document"

                if file_ext in ['.pdf'] or 'pdf' in content_type:
                    generated_doc_id = ingest_pdf(tmp_path, title=title_to_use)
                    logger.info(f"✅ Ingested PDF ({idx + 1}/{len(attachments_list)}): {upload.filename}")

                elif file_ext in ['.txt'] or 'text/plain' in content_type:
                    generated_doc_id = ingest_text_file(tmp_path, title=title_to_use)
                    logger.info(f"✅ Ingested text file ({idx + 1}/{len(attachments_list)}): {upload.filename}")

                elif file_ext in ['.png', '.jpg', '.jpeg'] or 'image' in content_type:
                    generated_doc_id = ingest_image(tmp_path, title=title_to_use)
                    logger.info(f"✅ Ingested image ({idx + 1}/{len(attachments_list)}): {upload.filename}")

                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {file_ext}. Supported: PDF, TXT, PNG, JPEG"
                    )

                if generated_doc_id:
                    uploaded_doc_ids.append(generated_doc_id)
                    attachment_metadata.append({
                        "filename": upload.filename,
                        "file_type": file_ext or content_type,
                        "doc_id": generated_doc_id,
                        "title": title_to_use
                    })
                    if doc_id is None:
                        doc_id = generated_doc_id  # Preserve first doc_id for backward compatibility

                    logger.info(f"📋 Document ID: {generated_doc_id}")
                    logger.info(f"Waiting for chunks for document {generated_doc_id}...")
                    try:
                        chunk_count = wait_for_chunks(generated_doc_id, max_wait_seconds=40)
                        logger.info(f"Found {chunk_count} chunks for {generated_doc_id}, ready to query")
                    except TimeoutError as e:
                        logger.warning(f"Timeout waiting for chunks for {generated_doc_id}: {e}. Proceeding anyway.")
            finally:
                if tmp_path and os.path.exists(tmp_path):
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
        # Priority: selected_doc_ids (explicit user selection) > doc_ids from upload
        # CRITICAL FIX: If user attaches documents, ALWAYS use them even if sidebar selection is empty
        doc_ids_to_use: Optional[List[str]] = None
        if selected_doc_ids_list is not None:
            # User explicitly provided selected_doc_ids (could be empty list)
            if len(selected_doc_ids_list) > 0:
                doc_ids_to_use = list(selected_doc_ids_list)  # Make a copy to avoid modifying original
                if uploaded_doc_ids:
                    for uploaded_doc_id in uploaded_doc_ids:
                        if uploaded_doc_id and uploaded_doc_id not in doc_ids_to_use:
                            doc_ids_to_use.append(uploaded_doc_id)
                    logger.info(f"Combining selected_doc_ids with uploaded doc_ids: {len(doc_ids_to_use)} document(s) total")
                else:
                    logger.info(f"Using selected_doc_ids: {len(doc_ids_to_use)} document(s)")
            else:
                # Empty selected_doc_ids list - check if user attached documents
                if uploaded_doc_ids:
                    # User attached documents but didn't select any from sidebar
                    # Use the uploaded documents (this is the intended behavior)
                    doc_ids_to_use = [doc for doc in uploaded_doc_ids if doc]
                    logger.info(f"selected_doc_ids is empty but user attached {len(doc_ids_to_use)} document(s) - using uploaded docs")
                else:
                    # No selected docs AND no uploaded docs - user truly has no documents
                    doc_ids_to_use = None
                    logger.info("selected_doc_ids is empty and no attachments - not filtering to any documents (user deselected all)")
        elif uploaded_doc_ids:
            doc_ids_to_use = [doc for doc in uploaded_doc_ids if doc]
            if doc_ids_to_use:
                logger.info(f"Using uploaded doc_ids for retrieval: {doc_ids_to_use}")
        elif doc_id:
            # Fallback to primary doc_id for backward compatibility when no uploads recorded
            doc_ids_to_use = [doc_id]
            logger.info(f"Using doc_id: {doc_id}...")

        doc_id_for_graph = doc_ids_to_use[0] if doc_ids_to_use else None
        
        # CRITICAL: Enable cross-doc mode for multi-document attachments
        # This makes multi-doc attachments follow the same successful path as cross-doc search
        # Thread B (cross-doc) succeeded with 91.8% confidence in 2 iterations
        # Thread A (3 attachments) failed with "I don't know" after 5 iterations
        # Solution: Treat multi-doc attachments like cross-doc search
        if doc_ids_to_use and len(doc_ids_to_use) > 1 and not cross_doc:
            cross_doc = True
            logger.info(f"🔄 Auto-enabling cross-doc mode for {len(doc_ids_to_use)} attached documents (following successful cross-doc strategy)")
        
        # Run the query with LangGraph
        if cross_doc:
            logger.info("Cross-document retrieval enabled")
        result = ask_with_graph(
            question,
            thread_id=thread_id_value,
            doc_id=doc_id_for_graph,  # Keep for backward compatibility
            selected_doc_ids=doc_ids_to_use,  # Use selected_doc_ids if provided
            cross_doc=cross_doc
        )
        
        # Get document title if doc_id is available
        final_doc_id = result.get("doc_id") or doc_id
        doc_title = None
        doc_titles_map: Dict[str, Optional[str]] = {}
        if final_doc_id:
            doc_title = get_document_title(final_doc_id)
            doc_titles_map[final_doc_id] = doc_title
        
        # Get doc_ids and pages from result
        doc_ids = result.get("doc_ids", [])
        pages = result.get("pages", [])
        
        # Use doc_map from citation_pruner if available (has "used" status)
        doc_map = result.get("doc_map", [])
        if doc_map:
            # Build doc_titles from doc_map (only used documents)
            doc_titles = [doc.get("title") for doc in doc_map if doc.get("used", False)]
            # Update doc_titles_map from doc_map
            for doc in doc_map:
                if doc.get("doc_id") and doc.get("title"):
                    doc_titles_map[doc["doc_id"]] = doc["title"]
            # Set final_doc_id if not already set
            if not final_doc_id and doc_ids:
                final_doc_id = doc_ids[0]
                doc_title = doc_titles_map.get(final_doc_id)
        else:
            # Fallback: Build doc_titles manually if doc_map not available
            if not final_doc_id and doc_ids:
                final_doc_id = doc_ids[0]
                doc_title = get_document_title(final_doc_id) if final_doc_id else None
                if final_doc_id:
                    doc_titles_map[final_doc_id] = doc_title
            
            # Collect titles for all reported doc_ids (preserving order)
            doc_titles: List[Optional[str]] = []
            if len(doc_ids) > 1:
                for doc_identifier in doc_ids:
                    if doc_identifier not in doc_titles_map:
                        doc_titles_map[doc_identifier] = get_document_title(doc_identifier)
                    doc_titles.append(doc_titles_map.get(doc_identifier))
        
        # Log thread interaction to database (synchronous operation, but FastAPI handles it)
        try:
            user_id_for_logging = user_id or "default_user"
            logger.info(f"infer_graph: Logging thread interaction with user_id='{user_id_for_logging}' (from Form user_id='{user_id}')")
            ingestion_meta = None
            if attachment_metadata:
                ingestion_meta = {
                    "attachments": attachment_metadata
                }
            record_id = log_thread_interaction(
                user_id=user_id_for_logging,
                thread_id=thread_id_value,
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
        
        # Don't return uploaded_doc_ids or doc_ids if action is abstain
        action = result.get("action", "answer")
        is_abstain = action == "abstain"
        
        # Get citations from citation_pruner if available
        citations = result.get("citations", [])
        
        response = {
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "action": action,
            "mode": "ingest_and_query" if attachments_list else "query_only",
            "pipeline": "langgraph",
            "thread_id": thread_id_value,
            "attachment_processed": len(attachments_list) > 0,
            "filenames": [meta["filename"] for meta in attachment_metadata] if attachment_metadata else None,
            "file_types": [meta["file_type"] for meta in attachment_metadata] if attachment_metadata else None,
            "uploaded_doc_ids": None if is_abstain else (uploaded_doc_ids if uploaded_doc_ids else None),
            "doc_id": None if is_abstain else final_doc_id,
            "doc_ids": [] if is_abstain else doc_ids,  # Clear doc_ids for abstain (already pruned by citation_pruner)
            "doc_title": None if is_abstain else doc_title,
            "doc_titles": [] if is_abstain else (doc_titles if doc_titles else None),
            "pages": [] if is_abstain else pages,  # Clear pages for abstain
            "cross_doc": cross_doc
        }
        
        # Add doc_map and citations from citation_pruner if available
        if doc_map and not is_abstain:
            response["doc_map"] = doc_map
        if citations and not is_abstain:
            response["citations"] = citations
        
        return response
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /infer-graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

