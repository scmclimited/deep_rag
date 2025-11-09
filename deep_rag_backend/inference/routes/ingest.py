"""
Ingest route - Ingest documents without querying.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from ingestion.ingest import ingest as ingest_pdf
from ingestion.ingest_text import ingest_text_file
from ingestion.ingest_image import ingest_image

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest")
async def ingest_endpoint(
    attachment: UploadFile = File(...),
    title: Optional[str] = Form(None)
):
    """
    Ingest a document into the vector database without querying.
    Supports PDF, TXT, PNG, JPEG.
    
    This endpoint only handles embedding and upsert - no agentic reasoning.
    Use /infer for ingestion + query with agentic reasoning.
    """
    try:
        logger.info(f"Starting ingestion for file: {attachment.filename}, size: {attachment.size if hasattr(attachment, 'size') else 'unknown'}")
        file_ext = Path(attachment.filename).suffix.lower() if attachment.filename else ""
        content_type = attachment.content_type or ""
        
        # Save uploaded file temporarily
        logger.info(f"Saving uploaded file temporarily...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await attachment.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        logger.info(f"File saved to temporary path: {tmp_path}")
        
        try:
            doc_id = None
            # Process based on file type
            logger.info(f"Processing file type: {file_ext} (content_type: {content_type})")
            if file_ext in ['.pdf'] or 'pdf' in content_type:
                logger.info(f"Starting PDF ingestion...")
                doc_id = ingest_pdf(tmp_path, title=title)
                logger.info(f"✅ Ingested PDF: {attachment.filename}")
                logger.info(f"📋 Document ID: {doc_id}")
                
            elif file_ext in ['.txt'] or 'text/plain' in content_type:
                logger.info(f"Starting text file ingestion...")
                doc_id = ingest_text_file(tmp_path, title=title or attachment.filename)
                logger.info(f"✅ Ingested text file: {attachment.filename}")
                logger.info(f"📋 Document ID: {doc_id}")
                
            elif file_ext in ['.png', '.jpg', '.jpeg'] or 'image' in content_type:
                logger.info(f"Starting image ingestion...")
                doc_id = ingest_image(tmp_path, title=title or attachment.filename)
                logger.info(f"✅ Ingested image: {attachment.filename}")
                logger.info(f"📋 Document ID: {doc_id}")
                
            else:
                logger.error(f"Unsupported file type: {file_ext} (content_type: {content_type})")
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}. Supported: PDF, TXT, PNG, JPEG"
                )
            
            logger.info(f"Ingestion completed successfully for {attachment.filename}, doc_id: {doc_id}")
            return {
                "status": "success",
                "filename": attachment.filename,
                "file_type": file_ext or content_type,
                "title": title or attachment.filename,
                "doc_id": doc_id
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /ingest endpoint: {e}", exc_info=True)
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error ingesting document: {str(e)}. Check server logs for details."
        )

