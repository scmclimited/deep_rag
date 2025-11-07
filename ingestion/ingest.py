"""
Main PDF ingestion module.

This module provides the main ingest function and maintains
backward compatibility by importing from modularized submodules.
"""
import os
import logging
import shutil
from pathlib import Path
from retrieval.db_utils import connect

from ingestion.pdf_extract import pdf_extract
from ingestion.chunking import semantic_chunks
from ingestion.db_ops import upsert_document, upsert_chunks
from ingestion.title_extract import extract_title

logger = logging.getLogger(__name__)


def ingest(pdf_path: str, title: str = None) -> str:
    """
    Ingest a PDF file into the vector database.
    
    Args:
        pdf_path: Path to PDF file
        title: Optional title for the document
        
    Returns:
        Document ID (UUID string)
        
    Raises:
        FileNotFoundError: If PDF file does not exist
    """
    # Resolve path - handle both relative and absolute paths
    # If path doesn't exist, try resolving relative to project root
    resolved_path = pdf_path
    if not os.path.isabs(pdf_path):
        # Try relative to current working directory first
        if not os.path.exists(pdf_path):
            # Try relative to project root (assuming we're in /app in Docker)
            project_root = os.getenv("PROJECT_ROOT", os.getcwd())
            alt_path = os.path.join(project_root, pdf_path.lstrip("/"))
            if os.path.exists(alt_path):
                resolved_path = alt_path
        else:
            resolved_path = os.path.abspath(pdf_path)
    else:
        resolved_path = pdf_path
    
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}. Tried: {resolved_path}")
    
    logger.info(f"Starting ingestion for: {resolved_path}")
    pages = pdf_extract(resolved_path, extract_images=True)
    logger.info(f"Extracted {len(pages)} pages from PDF (with images)")
    
    chunks, temp_dir = semantic_chunks(pages)
    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    
    # Extract title
    final_title = extract_title(resolved_path, pages, title)
    
    doc_id = None
    try:
        with connect() as conn, conn.cursor() as cur:
            doc_id = upsert_document(cur, final_title, resolved_path)
            logger.info(f"Document inserted: doc_id={doc_id}, title={final_title}")
            
            upsert_chunks(cur, doc_id, chunks, temp_dir)
            
            conn.commit()
            logger.info(f"Ingestion complete: doc_id={doc_id}, {len(chunks)} chunks stored")
    finally:
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
    
    print(f"Ingested: {resolved_path} (title: {final_title}, {len(chunks)} chunks)")
    return doc_id


if __name__ == "__main__":
    import sys
    ingest(sys.argv[1], title=None if len(sys.argv) < 3 else sys.argv[2])
