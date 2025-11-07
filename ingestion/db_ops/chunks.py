"""
Chunk database operations.
"""
import logging
from uuid import uuid4
from typing import List, Tuple, Optional
import psycopg2.extras as pe

from ingestion.embeddings import embed_text, embed_image, embed_multi_modal

logger = logging.getLogger(__name__)


def upsert_chunks(cur, doc_id: str, chunks: List[Tuple], temp_dir: Optional[str] = None) -> None:
    """
    Insert chunks into database with multi-modal embeddings.
    
    Chunks format: (text, page_start, page_end, is_ocr, is_figure, content_type, image, image_path)
    
    Args:
        cur: Database cursor
        doc_id: Document ID
        chunks: List of chunk tuples
        temp_dir: Optional temporary directory path for cleanup
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

