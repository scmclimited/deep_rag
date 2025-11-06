# ingest_image.py
# Handle image file ingestion (PNG, JPEG) using OCR and optional vision models
from uuid import uuid4
from pathlib import Path
import os
import logging
from psycopg2 import connect
from dotenv import load_dotenv
import numpy as np
import psycopg2, psycopg2.extras as pe

try:
    from PIL import Image
    import pytesseract
    from pdf2image import convert_from_path
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False

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

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from image using OCR.
    For future: Could add vision-language model (CLIP, BLIP) for image understanding.
    """
    if not IMAGE_AVAILABLE:
        raise ImportError("Image processing libraries not available. Install: pip install Pillow pytesseract pdf2image")
    
    try:
        # Open image
        image = Image.open(image_path)
        
        # Extract text using OCR
        text = pytesseract.image_to_string(image, lang='eng')
        
        # If OCR returns very little text, add a note about the image
        if len(text.strip()) < 50:
            text = f"[Image file: {Path(image_path).name}]\n[OCR extracted text: {text.strip()}]\n[Note: Image may contain visual content not captured by text]"
        
        return text.strip()
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}. Using fallback description.")
        return f"[Image file: {Path(image_path).name}]\n[OCR extraction failed. Image may contain visual content.]"

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
    logger.info(f"Upserting {len(chunks)} chunks for image document {doc_id}")
    
    for chunk_index, chunk_data in enumerate(chunks):
        try:
            # Handle both old format (5-tuple) and new format (7-tuple)
            if len(chunk_data) == 5:
                text, p0, p1, is_ocr, is_fig = chunk_data
                content_type = 'image'
                image_path = None
            else:
                text, p0, p1, is_ocr, is_fig, content_type, image_path = chunk_data
            
            # For images: embed text (OCR) + image together using CLIP
            try:
                if image_path and os.path.exists(image_path):
                    from PIL import Image
                    image = Image.open(image_path).convert('RGB')
                    emb = embed_multi_modal(text=text, image_path=image, normalize_emb=True)
                else:
                    # Text-only embedding (OCR text)
                    emb = embed_text(text, normalize_emb=True)
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {chunk_index}: {e}", exc_info=True)
                logger.warning(f"Skipping chunk {chunk_index} due to embedding failure")
                continue
            
            cid = str(uuid4())
            
            cur.execute("""
              INSERT INTO chunks (chunk_id, doc_id, page_start, page_end, section, text, is_ocr, is_figure, content_type, image_path, lex, emb, meta)
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, to_tsvector('simple', unaccent(%s)), %s, %s)
            """, (cid, doc_id, p0, p1, None, text, is_ocr, is_fig, 'image', image_path, text, emb.tolist(), pe.Json({"len": len(text), "source": "image"})))
            
        except Exception as e:
            logger.error(f"Failed to insert chunk {chunk_index}: {e}", exc_info=True)
            raise
    
    logger.info(f"Successfully upserted {len(chunks)} chunks for image document {doc_id}")

def ingest_image(image_path: str, title: str = None):
    """
    Ingest an image file (PNG, JPEG) into the vector database.
    
    Current implementation:
    - Extracts text via OCR (Tesseract)
    - Uses text-only embeddings (BAAI/bge-m3)
    
    Future enhancement:
    - Add CLIP-style vision-language embeddings for true multi-modal search
    - Store both text and image embeddings
    """
    if not IMAGE_AVAILABLE:
        raise ImportError("Image processing libraries not available. Install: pip install Pillow pytesseract")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Determine title
    if not title:
        title = Path(image_path).stem
    
    logger.info(f"Starting image ingestion: {image_path}")
    
    # Extract text from image
    text = extract_text_from_image(image_path)
    
    if not text.strip():
        logger.warning(f"No text extracted from image: {image_path}")
        text = f"[Image: {Path(image_path).name}]"
    
    # Create chunks for the image
    chunks = []
    # For images, we'll create a single chunk with text + image
    # The image path is the input path
    image_path_to_use = image_path
    
    if len(text) > 2000:
        # Split long OCR text into chunks
        import re
        parts = re.split(r'[.!?]\s+', text)
        current_chunk = []
        current_len = 0
        
        for part in parts:
            part_len = len(part.split())
            # CLIP-compatible: max 25 words per chunk (≈32-37 tokens, safe margin for 77 limit)
            if current_len + part_len > 25 and current_chunk:
                chunks.append((" ".join(current_chunk), 1, 1, True, False, 'image', image_path_to_use))
                current_chunk = [part]
                current_len = part_len
            else:
                current_chunk.append(part)
                current_len += part_len
        
        if current_chunk:
            chunks.append((" ".join(current_chunk), 1, 1, True, False, 'image', image_path_to_use))
    else:
        # Single chunk with image
        chunks.append((text, 1, 1, True, False, 'image', image_path_to_use))  # (text, page_start, page_end, is_ocr=True, is_figure=False, content_type, image_path)
    
    logger.info(f"Created {len(chunks)} chunks from image")
    
    # Insert into database
    with connect() as conn, conn.cursor() as cur:
        doc_id = upsert_document(cur, title, image_path)
        logger.info(f"Document inserted: doc_id={doc_id}, title={title}")
        
        upsert_chunks(cur, doc_id, chunks)
        
        conn.commit()
        logger.info(f"Ingestion complete: doc_id={doc_id}, {len(chunks)} chunks stored")
    
    print(f"Ingested: {image_path} (title: {title}, {len(chunks)} chunks)")

