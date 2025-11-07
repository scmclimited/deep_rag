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

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/infer-graph")
async def infer_graph(
    question: str = Form(...),
    attachment: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    thread_id: Optional[str] = Form("default"),
    cross_doc: bool = Form(False)
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
        
        # Run the query with LangGraph, passing doc_id for document-specific retrieval
        if cross_doc:
            logger.info("Cross-document retrieval enabled")
        answer = ask_with_graph(question, thread_id=thread_id, doc_id=doc_id, cross_doc=cross_doc)
        
        return {
            "answer": answer,
            "mode": "ingest_and_query" if attachment else "query_only",
            "pipeline": "langgraph",
            "thread_id": thread_id,
            "attachment_processed": attachment is not None,
            "filename": attachment.filename if attachment else None,
            "file_type": file_ext or content_type if attachment else None,
            "doc_id": doc_id,
            "cross_doc": cross_doc
        }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /infer-graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

