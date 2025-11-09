"""
Infer route - Combined ingestion and query using direct pipeline.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from inference.agents import run_deep_rag
from ingestion.ingest import ingest as ingest_pdf
from ingestion.ingest_text import ingest_text_file
from ingestion.ingest_image import ingest_image
from retrieval.retrieval import wait_for_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/infer")
async def infer(
    question: str = Form(...),
    attachment: UploadFile = File(None),
    title: Optional[str] = Form(None),
    cross_doc: bool = Form(False)
):
    """
    Combined ingestion and query endpoint using direct pipeline.
    
    If attachment is provided:
    - Ingest the attachment (PDF, TXT, PNG, JPEG)
    - Then query using the question
    
    If no attachment:
    - Just query existing documents (same as /ask)
    
    Uses inference/agents/pipeline.py (direct pipeline without LangGraph).
    
    Supported file types:
    - PDF: Full text extraction with OCR fallback
    - TXT: Direct text ingestion
    - PNG/JPEG: Image captioning (extracts text via OCR/vision)
    """
    try:
        # If no attachment, just query existing documents
        if not attachment:
            if cross_doc:
                logger.info("Cross-document retrieval enabled")
            answer = run_deep_rag(question, cross_doc=cross_doc)
            return {
                "answer": answer,
                "mode": "query_only",
                "pipeline": "direct",
                "attachment_processed": False,
                "cross_doc": cross_doc
            }
        
        # Process attachment
        file_ext = Path(attachment.filename).suffix.lower() if attachment.filename else ""
        content_type = attachment.content_type or ""
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await attachment.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            doc_id = None
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
            
            # After ingestion, run the query with doc_id filter for document-specific retrieval
            if cross_doc:
                logger.info("Cross-document retrieval enabled")
            answer = run_deep_rag(question, doc_id=doc_id, cross_doc=cross_doc)
            
            return {
                "answer": answer,
                "mode": "ingest_and_query",
                "pipeline": "direct",
                "attachment_processed": True,
                "filename": attachment.filename,
                "file_type": file_ext or content_type,
                "doc_id": doc_id,
                "cross_doc": cross_doc
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /infer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

