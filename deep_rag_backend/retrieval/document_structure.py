"""
Document structure-based retrieval for explicit document selection.
When a document is explicitly selected and query is ambiguous, retrieve by document structure
instead of similarity matching.
"""
import logging
from typing import List, Dict, Optional
from retrieval.db_utils import connect
from retrieval.vector_utils import parse_vector

logger = logging.getLogger(__name__)


def retrieve_by_document_structure(
    doc_id: str,
    max_chunks: int = 20,
    strategy: str = "first_pages"
) -> List[Dict]:
    """
    Retrieve chunks from a document by structure (not similarity).
    
    Useful when:
    - Document is explicitly selected
    - Query is ambiguous (e.g., "share the details of this document")
    - Similarity-based retrieval returns few/poor results
    
    Args:
        doc_id: Document ID to retrieve from
        max_chunks: Maximum number of chunks to return
        strategy: Retrieval strategy
            - "first_pages": Retrieve from first N pages (good for overview)
            - "all_pages": Retrieve chunks from all pages (distributed)
            - "sequential": Retrieve chunks in page order
        
    Returns:
        List of chunks with structure-based ordering
    """
    if not doc_id:
        return []
    
    logger.info(f"Structure-based retrieval for document {doc_id[:8]}... (strategy: {strategy})")
    
    try:
        with connect() as conn, conn.cursor() as cur:
            if strategy == "first_pages":
                # Get chunks from first 10 pages, ordered by page number
                cur.execute("""
                    SELECT 
                        chunk_id, doc_id, text, page_start, page_end,
                        COALESCE(content_type, 'text') as content_type,
                        COALESCE(image_path, '') as image_path,
                        emb
                    FROM chunks
                    WHERE doc_id = %s
                        AND page_start IS NOT NULL
                        AND page_start <= 10
                    ORDER BY page_start, page_end, chunk_id
                    LIMIT %s
                """, (doc_id, max_chunks))
            elif strategy == "all_pages":
                # Get chunks distributed across all pages
                cur.execute("""
                    SELECT 
                        chunk_id, doc_id, text, page_start, page_end,
                        COALESCE(content_type, 'text') as content_type,
                        COALESCE(image_path, '') as image_path,
                        emb
                    FROM chunks
                    WHERE doc_id = %s
                    ORDER BY page_start, page_end, chunk_id
                    LIMIT %s
                """, (doc_id, max_chunks))
            else:  # sequential
                # Get chunks in sequential order
                cur.execute("""
                    SELECT 
                        chunk_id, doc_id, text, page_start, page_end,
                        COALESCE(content_type, 'text') as content_type,
                        COALESCE(image_path, '') as image_path,
                        emb
                    FROM chunks
                    WHERE doc_id = %s
                    ORDER BY page_start, page_end, chunk_id
                    LIMIT %s
                """, (doc_id, max_chunks))
            
            rows = cur.fetchall()
            
            if not rows:
                logger.warning(f"No chunks found for document {doc_id[:8]}...")
                return []
            
            # Convert to chunk dict format (matching retrieve_hybrid output)
            chunks = []
            for row in rows:
                chunk_id, doc_id_val, text, p0, p1, content_type, image_path, emb = row
                
                # Parse embedding if available
                emb_parsed = parse_vector(emb) if emb else None
                
                chunk = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id_val,
                    "text": text or "",
                    "p0": p0,
                    "p1": p1,
                    "content_type": content_type,
                    "image_path": image_path,
                    # Structure-based retrieval doesn't have similarity scores
                    # Set default scores so they don't break downstream processing
                    "lex": 0.5,  # Neutral score
                    "vec": 0.5,  # Neutral score
                    "ce": 0.0,  # No reranker score
                }
                
                if emb_parsed is not None:
                    chunk["emb"] = emb_parsed
                
                chunks.append(chunk)
            
            logger.info(f"Retrieved {len(chunks)} chunks via structure-based retrieval")
            if chunks:
                pages = sorted(set([c["p0"] for c in chunks if c["p0"] is not None]))
                logger.info(f"Pages represented: {pages[:10]}{'...' if len(pages) > 10 else ''}")
            
            return chunks
            
    except Exception as e:
        logger.error(f"Structure-based retrieval failed: {e}", exc_info=True)
        return []

