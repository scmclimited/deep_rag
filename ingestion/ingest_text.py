# ingest_text.py
# Handle plain text file ingestion
from uuid import uuid4
from pathlib import Path
import os
import logging
from psycopg2 import connect
from dotenv import load_dotenv
import numpy as np
import psycopg2, psycopg2.extras as pe

load_dotenv()

logger = logging.getLogger(__name__)

# Use unified multi-modal embedding system (CLIP)
from ingestion.embeddings import embed_text, embed_image, embed_multi_modal, normalize, EMBEDDING_DIM

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)

def connect():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        dbname=os.getenv("DB_NAME")
    )

def semantic_chunks_text(text: str, max_words=25, overlap=5):
    """
    Split text into semantic chunks with overlap.
    Similar to PDF chunking but for plain text.
    
    Note: CLIP-ViT-B/32 has max 77 tokens per text. 
    We use max_words=25 (words) to ensure chunks stay well under 77 tokens.
    Average word-to-token ratio is ~1.3-1.5, so 25 words ≈ 32-37 tokens (safe margin).
    """
    import re
    chunks = []
    
    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    buf, count = [], 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        toks = len(para.split())  # Word count, not token count
        if count + toks > max_words and buf:
            chunk_text = " ".join(buf)
            chunks.append((chunk_text, 1, 1, False, False))  # (text, page_start, page_end, is_ocr, is_figure)
            
            # Calculate overlap
            overlap_words = " ".join(chunk_text.split()[-overlap:])
            buf, count = [overlap_words, para], len(overlap_words.split()) + toks
        else:
            buf.append(para)
            count += toks
    
    if buf:
        chunk_text = " ".join(buf)
        chunks.append((chunk_text, 1, 1, False, False))
    
    return chunks

def upsert_document(cur, title, path):
    """Insert or update document record."""
    did = uuid4()
    cur.execute(
        "INSERT INTO documents (doc_id, title, source_path) VALUES (%s,%s,%s)",
        (str(did), title, path)
    )
    return str(did)

def upsert_chunks(cur, doc_id, chunks):
    """Insert chunks with embeddings."""
    logger.info(f"Upserting {len(chunks)} chunks for document {doc_id}")
    
    for chunk_index, (text, p0, p1, is_ocr, is_fig) in enumerate(chunks):
        try:
            # Generate embedding - skip chunk if embedding fails
            try:
                emb = embed_text(text, normalize_emb=True)
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {chunk_index}: {e}", exc_info=True)
                logger.warning(f"Skipping chunk {chunk_index} due to embedding failure")
                continue
            
            cid = str(uuid4())
            
            cur.execute("""
              INSERT INTO chunks (chunk_id, doc_id, page_start, page_end, section, text, is_ocr, is_figure, content_type, lex, emb, meta)
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, to_tsvector('simple', unaccent(%s)), %s, %s)
            """, (cid, doc_id, p0, p1, None, text, is_ocr, is_fig, 'text', text, emb.tolist(), pe.Json({"len": len(text), "content_type": "text"})))
            
        except Exception as e:
            logger.error(f"Failed to insert chunk {chunk_index}: {e}", exc_info=True)
            raise
    
    logger.info(f"Successfully upserted {len(chunks)} chunks for document {doc_id}")

def ingest_text_file(text_path: str, title: str = None):
    """
    Ingest a plain text file into the vector database.
    """
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file not found: {text_path}")
    
    # Read text file
    with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    if not text.strip():
        raise ValueError(f"Text file is empty: {text_path}")
    
    # Determine title
    if not title:
        title = Path(text_path).stem
    
    logger.info(f"Starting text file ingestion: {text_path}")
    
    # Chunk the text
    chunks = semantic_chunks_text(text)
    logger.info(f"Created {len(chunks)} chunks from text file")
    
    # Insert into database
    with connect() as conn, conn.cursor() as cur:
        doc_id = upsert_document(cur, title, text_path)
        logger.info(f"Document inserted: doc_id={doc_id}, title={title}")
        
        upsert_chunks(cur, doc_id, chunks)
        
        conn.commit()
        logger.info(f"Ingestion complete: doc_id={doc_id}, {len(chunks)} chunks stored")
    
    print(f"Ingested: {text_path} (title: {title}, {len(chunks)} chunks)")

