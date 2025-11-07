"""
Inspect document chunks and pages in the database.
"""
import logging
from retrieval.db_utils import connect

logger = logging.getLogger(__name__)


def inspect_document(doc_title: str = None, doc_id: str = None):
    """
    Inspect all chunks stored for a document, showing page distribution and sample text.
    
    Args:
        doc_title: Document title to search for (partial match)
        doc_id: Document ID (UUID) - if provided, doc_title is ignored
        
    Returns:
        Dictionary with document info and chunk statistics
    """
    with connect() as conn, conn.cursor() as cur:
        # Find document
        if doc_id:
            cur.execute("SELECT doc_id, title, source_path FROM documents WHERE doc_id = %s", (doc_id,))
        elif doc_title:
            cur.execute("SELECT doc_id, title, source_path FROM documents WHERE title ILIKE %s", (f"%{doc_title}%",))
        else:
            # Get all documents
            cur.execute("SELECT doc_id, title, source_path FROM documents ORDER BY created_at DESC LIMIT 10")
            rows = cur.fetchall()
            if not rows:
                return {"error": "No documents found. Use doc_title or doc_id to specify a document."}
            print("Available documents:")
            for doc_id, title, path in rows:
                print(f"  - {title} (doc_id: {doc_id}, path: {path})")
            return {"documents": [{"doc_id": doc_id, "title": title, "source_path": path} for doc_id, title, path in rows]}
        
        doc_row = cur.fetchone()
        if not doc_row:
            return {"error": f"Document not found: {doc_title or doc_id}"}
        
        doc_id, title, source_path = doc_row
        
        # Get all chunks for this document
        cur.execute("""
            SELECT 
                chunk_id, page_start, page_end, 
                content_type, 
                LENGTH(text) as text_length,
                LEFT(text, 200) as text_preview,
                is_ocr, is_figure
            FROM chunks 
            WHERE doc_id = %s
            ORDER BY page_start, page_end, chunk_id
        """, (doc_id,))
        
        chunks = cur.fetchall()
        
        # Analyze page distribution
        pages = {}
        for chunk_id, p0, p1, content_type, text_len, text_preview, is_ocr, is_fig in chunks:
            page_key = f"{p0}-{p1}"
            if page_key not in pages:
                pages[page_key] = {
                    "page_start": p0,
                    "page_end": p1,
                    "chunk_count": 0,
                    "total_text_length": 0,
                    "chunks": []
                }
            pages[page_key]["chunk_count"] += 1
            pages[page_key]["total_text_length"] += text_len or 0
            pages[page_key]["chunks"].append({
                "chunk_id": str(chunk_id)[:8] + "...",
                "content_type": content_type,
                "text_length": text_len,
                "text_preview": text_preview,
                "is_ocr": is_ocr,
                "is_figure": is_fig
            })
        
        # Get summary statistics
        cur.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT page_start) as unique_pages,
                MIN(page_start) as min_page,
                MAX(page_end) as max_page,
                AVG(LENGTH(text)) as avg_text_length
            FROM chunks
            WHERE doc_id = %s
        """, (doc_id,))
        
        stats = cur.fetchone()
        
        result = {
            "document": {
                "doc_id": doc_id,
                "title": title,
                "source_path": source_path
            },
            "statistics": {
                "total_chunks": stats[0],
                "unique_pages": stats[1],
                "page_range": f"{stats[2]}-{stats[3]}" if stats[2] and stats[3] else "N/A",
                "avg_text_length": int(stats[4]) if stats[4] else 0
            },
            "page_distribution": pages
        }
        
        return result

