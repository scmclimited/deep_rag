# ingest.py
from uuid import uuid4
from pathlib import Path
import os
import logging
from psycopg2 import connect
from dotenv import load_dotenv
import fitz  # PyMuPDF (robust text/blocks); alternative: pdfplumber
from pdf2image import convert_from_path
import pytesseract
import psycopg2, psycopg2.extras as pe
import numpy as np
import re
from PIL import Image
import io
import tempfile

# Use unified multi-modal embedding system
from ingestion.embeddings import embed_text, embed_image, embed_multi_modal, normalize, EMBEDDING_DIM

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reranker for query time (still uses text-only cross-encoder)
# Note: CrossEncoder is imported here (not used in ingestion, but kept for consistency)
RERANK_MODEL = "BAAI/bge-reranker-base"
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(RERANK_MODEL)
except Exception as e:
    logger.warning(f"Reranker not available: {e}. Continuing without reranking.")
    reranker = None

def pdf_extract(path: str, extract_images: bool = True):
    """
    Extract pagewise text, OCR fallback if blank; extract images if present.
    
    Returns:
        List of page dicts with:
        - page: page number
        - text: extracted text
        - images: list of PIL Images found on page
        - captions: figure captions
        - is_ocr: whether OCR was used
    """
    doc = fitz.open(path)
    pages = []
    
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        text = re.sub(r'[ \t]+', ' ', text).strip()
        is_scan = (len(text) < 20)
        ocr_text = ""
        if is_scan:
            # OCR at page-level
            images = convert_from_path(path, first_page=i+1, last_page=i+1, dpi=300)
            ocr_text = pytesseract.image_to_string(images[0])
        final_text = text if len(text) >= len(ocr_text) else ocr_text

        # Extract images from PDF page
        page_images = []
        if extract_images:
            try:
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                        page_images.append(image)
                        logger.debug(f"Extracted image {img_index} from page {i+1}")
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {i+1}: {e}")
            except Exception as e:
                logger.warning(f"Failed to extract images from page {i+1}: {e}")

        # Extract figure captions
        captions = []
        for block in page.get_text("blocks"):
            btxt = block[4].strip()
            if re.search(r'^(Figure|Fig\.|Diagram)\s*\d+', btxt, re.I):
                captions.append(btxt)

        pages.append({
            "page": i+1,
            "text": final_text,
            "images": page_images,
            "captions": captions,
            "is_ocr": is_scan and len(final_text) > 0
        })
    
    doc.close()
    return pages

def semantic_chunks(page_items, max_words=25, overlap=5):
    """
    Split pages into chunks with multi-modal support.
    
    Note: CLIP-ViT-B/32 has max 77 tokens per text. 
    We use max_words=25 (words) to ensure chunks stay well under 77 tokens.
    Average word-to-token ratio is ~1.3-1.5, so 25 words ≈ 32-37 tokens (safe margin).
    
    Returns list of tuples:
    (text, page_start, page_end, is_ocr, is_figure, content_type, image, image_path)
    
    content_type: 'text', 'image', 'multimodal', 'pdf_text', 'pdf_image'
    image: PIL Image or None
    image_path: str path to saved image or None
    """
    chunks = []
    chunk_index = 0
    logger.info(f"Starting chunking with max_words={max_words} (CLIP-compatible, ~{int(max_words * 1.4)} tokens), overlap={overlap}")
    
    # Create temp directory for saving images
    temp_dir = tempfile.mkdtemp(prefix="pdf_images_")
    
    for p in page_items:
        page_num = p["page"]
        text = p["text"]
        images = p.get("images", [])
        captions = p.get("captions", [])
        is_ocr = p.get("is_ocr", False)
        
        # Process text chunks
        units = re.split(r'(?m)^(#+\s.*|[A-Z][^\n]{0,80}\n[-=]{3,}\s*$)|\n{2,}', text)
        units = [u for u in units if u and u.strip()]
        buf, count = [], 0
        
        for u in units:
            toks = len(u.split())  # Word count, not token count
            if count + toks > max_words and buf:
                chunk_text = " ".join(buf)
                content_type = 'pdf_text' if text else 'text'
                
                # Check if there are images on this page that might relate to this chunk
                # If images exist, we'll create multimodal chunks
                if images:
                    # For multimodal: combine text with first image
                    image = images[0] if images else None
                    image_path = None
                    if image:
                        # Save image temporarily
                        image_path = os.path.join(temp_dir, f"page_{page_num}_img_0.png")
                        image.save(image_path)
                        content_type = 'multimodal'
                    
                    chunks.append((chunk_text, page_num, page_num, is_ocr, False, content_type, image, image_path))
                else:
                    chunks.append((chunk_text, page_num, page_num, is_ocr, False, content_type, None, None))
                
                chunk_index += 1
                overlap_words = " ".join(chunk_text.split()[-overlap:])
                buf, count = [overlap_words, u], len(overlap_words.split()) + toks
            else:
                buf.append(u)
                count += toks
                
        # Final text chunk
        if buf:
            chunk_text = " ".join(buf)
            content_type = 'pdf_text' if text else 'text'
            
            if images:
                image = images[0] if images else None
                image_path = None
                if image:
                    image_path = os.path.join(temp_dir, f"page_{page_num}_img_final.png")
                    image.save(image_path)
                    content_type = 'multimodal'
                chunks.append((chunk_text, page_num, page_num, is_ocr, False, content_type, image, image_path))
            else:
                chunks.append((chunk_text, page_num, page_num, is_ocr, False, content_type, None, None))
            chunk_index += 1

        # Create separate chunks for images (if not already combined with text)
        for img_idx, image in enumerate(images):
            if len(images) > 1 or not text:  # Create separate image chunk if multiple images or no text
                image_path = os.path.join(temp_dir, f"page_{page_num}_img_{img_idx}.png")
                image.save(image_path)
                # Use caption if available, otherwise use image description
                caption_text = captions[img_idx] if img_idx < len(captions) else f"[Image {img_idx + 1} from page {page_num}]"
                chunks.append((caption_text, page_num, page_num, is_ocr, True, 'pdf_image', image, image_path))
                chunk_index += 1

        # Add caption chunks
        for cap in captions:
            chunks.append((cap, page_num, page_num, is_ocr, True, 'text', None, None))
            chunk_index += 1
    
    logger.info(f"Chunking complete: created {len(chunks)} chunks total")
    return chunks, temp_dir

def connect():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        dbname=os.getenv("DB_NAME")
    )

def upsert_document(cur, title, path):
    did = uuid4()
    cur.execute(
        "INSERT INTO documents (doc_id, title, source_path) VALUES (%s,%s,%s)",
        (str(did), title, path)
    )
    return str(did)

def upsert_chunks(cur, doc_id, chunks, temp_dir=None):
    """
    Insert chunks into database with multi-modal embeddings.
    
    Chunks format: (text, page_start, page_end, is_ocr, is_figure, content_type, image, image_path)
    """
    logger.info(f"Upserting {len(chunks)} chunks for document {doc_id}")
    
    for chunk_index, chunk_data in enumerate(chunks):
        try:
            # Handle both old format (5-tuple) and new format (8-tuple)
            if len(chunk_data) == 5:
                text, p0, p1, is_ocr, is_fig = chunk_data
                content_type = 'text'
                image = None
                image_path = None
            else:
                text, p0, p1, is_ocr, is_fig, content_type, image, image_path = chunk_data
            
            # Generate embedding based on content type
            try:
                if content_type in ['multimodal', 'pdf_image'] and image is not None:
                    # Multi-modal: embed text + image together
                    emb = embed_multi_modal(text=text, image_path=image, normalize_emb=True)
                elif content_type == 'image' and image is not None:
                    # Image only
                    emb = embed_image(image, normalize_emb=True)
                else:
                    # Text only
                    emb = embed_text(text, normalize_emb=True)
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {chunk_index}: {e}", exc_info=True)
                # Skip this chunk if embedding fails - don't insert corrupted data
                logger.warning(f"Skipping chunk {chunk_index} due to embedding failure")
                continue
            
            cid = str(uuid4())
            
            # Insert into database with multi-modal support
            cur.execute("""
              INSERT INTO chunks (
                chunk_id, doc_id, page_start, page_end, section, text, 
                is_ocr, is_figure, content_type, image_path,
                lex, emb, meta
              )
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, to_tsvector('simple', unaccent(%s)), %s, %s)
            """, (
                cid, doc_id, p0, p1, None, text,
                is_ocr, is_fig, content_type, image_path,
                text, emb.tolist(), 
                pe.Json({"len": len(text), "content_type": content_type})
            ))
            
            logger.info(
                f"Chunk {chunk_index} inserted: chunk_id={cid}, "
                f"pages={p0}-{p1}, content_type={content_type}, "
                f"tokens={len(text.split()) if text else 0}, "
                f"has_image={image is not None}",
                extra={
                    "chunk_index": chunk_index,
                    "chunk_id": cid,
                    "doc_id": doc_id,
                    "page_start": p0,
                    "page_end": p1,
                    "content_type": content_type,
                    "token_count": len(text.split()) if text else 0,
                    "character_count": len(text) if text else 0,
                    "is_ocr": is_ocr,
                    "is_figure": is_fig,
                    "has_image": image is not None
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to insert chunk {chunk_index}: {e}",
                exc_info=True,
                extra={
                    "chunk_index": chunk_index,
                    "error": str(e)
                }
            )
            raise
    
    logger.info(f"Successfully upserted {len(chunks)} chunks for document {doc_id}")
    
    # Clean up temp directory if provided
    if temp_dir and os.path.exists(temp_dir):
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")

def ingest(pdf_path: str, title: str=None):
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
    
    # Determine title: use provided title, or extract from PDF, or use filename
    if title:
        # Truncate provided title to 50 words max to avoid token issues
        words = title.split()
        if len(words) > 20:
            final_title = " ".join(words[:20])
            print(f"Provided title truncated from {len(words)} words to 20 words")
        else:
            final_title = title
        print(f"Using provided title: {final_title}")
    else:
        # Try to extract title from PDF metadata or first page
        try:
            doc = fitz.open(resolved_path)
            metadata = doc.metadata
            if metadata.get("title"):
                # Truncate metadata title to 20 words max
                metadata_title = metadata["title"]
                words = metadata_title.split()
                if len(words) > 20:
                    final_title = " ".join(words[:50])
                    print(f"Metadata title truncated from {len(words)} words to 50 words")
                else:
                    final_title = metadata_title
                print(f"Extracted title from PDF metadata: {final_title}")
            elif pages and pages[0].get("text"):
                # Extract first line or first 100 chars as title
                # Truncate to 50 words max to avoid CLIP token issues if title is embedded elsewhere
                first_text = pages[0]["text"].strip()
                first_line = first_text.split("\n")[0][:100].strip()
                # Further truncate to 50 words if it's very long
                words = first_line.split()
                if len(words) > 20:
                    first_line = " ".join(words[:20])
                    print(f"Title truncated from {len(words)} words to 20 words")
                final_title = first_line if first_line else Path(resolved_path).stem
                if first_line:
                    print(f"Extracted title from first page: {final_title}")
                else:
                    print(f"Using filename as title: {final_title}")
            else:
                final_title = Path(resolved_path).stem  # filename without extension
                print(f"Using filename as title: {final_title}")
            doc.close()
        except Exception as e:
            final_title = Path(resolved_path).stem
            print(f"Could not extract title from PDF, using filename: {final_title} (error: {e})")
    
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
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
    
    print(f"Ingested: {resolved_path} (title: {final_title}, {len(chunks)} chunks)")
    return doc_id

if __name__ == "__main__":
    import sys
    ingest(sys.argv[1], title=None if len(sys.argv)<3 else sys.argv[2])
