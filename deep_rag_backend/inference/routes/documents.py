"""
Documents route - List and manage documents.
"""
import logging
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, Query
from retrieval.db_utils import connect

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/documents")
def get_documents(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of documents to return")
) -> Dict[str, List[Dict]]:
    """
    Get list of all documents in the database.
    
    Returns:
        Dictionary with 'documents' list containing doc_id, title, source_path, created_at
    """
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, title, source_path, created_at
                FROM documents
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,)
            )
            rows = cur.fetchall()
            
            documents = []
            for row in rows:
                doc_id, title, source_path, created_at = row
                documents.append({
                    "doc_id": str(doc_id),
                    "title": title or "Untitled",
                    "source_path": source_path,
                    "created_at": created_at.isoformat() if created_at else None
                })
            
            return {"documents": documents}
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> Dict[str, str]:
    """
    Delete a document and all its associated chunks.
    
    Args:
        doc_id: Document ID (UUID) to delete
        
    Returns:
        Success message with deleted doc_id
    """
    try:
        with connect() as conn, conn.cursor() as cur:
            # Check if document exists
            cur.execute(
                "SELECT doc_id, title FROM documents WHERE doc_id = %s",
                (doc_id,)
            )
            doc = cur.fetchone()
            
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            
            # Delete document (chunks will be deleted via CASCADE)
            cur.execute(
                "DELETE FROM documents WHERE doc_id = %s",
                (doc_id,)
            )
            
            # Also delete diagnostic reports if they exist
            import os
            from pathlib import Path
            diagnostic_reports_dir = Path("retrieval/diagnostics/diagnostic_reports")
            if diagnostic_reports_dir.exists():
                # Find and delete diagnostic reports for this doc_id
                for pattern in [f"*doc_id={doc_id}.json", f"*_doc_id={doc_id}.json"]:
                    for report_file in diagnostic_reports_dir.glob(pattern):
                        try:
                            os.remove(report_file)
                            logger.info(f"Deleted diagnostic report: {report_file}")
                        except Exception as e:
                            logger.warning(f"Failed to delete diagnostic report {report_file}: {e}")
            
            conn.commit()
            
            logger.info(f"Deleted document: doc_id={doc_id}, title={doc[1]}")
            
            return {
                "message": "Document deleted successfully",
                "doc_id": doc_id,
                "title": doc[1]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

