"""
Health check route.
"""
import logging
from fastapi import APIRouter, HTTPException
from retrieval.db_utils import connect

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health():
    """
    Health check endpoint that verifies:
    - API is running
    - Database connection is available
    - Required tables exist (documents, chunks, thread_tracking)
    """
    try:
        # Check database connection
        with connect() as conn, conn.cursor() as cur:
            # Verify required tables exist
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('documents', 'chunks', 'thread_tracking')
                ORDER BY table_name
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            required_tables = ['documents', 'chunks', 'thread_tracking']
            missing_tables = set(required_tables) - set(tables)
            
            if missing_tables:
                logger.warning(f"Missing required tables: {missing_tables}")
                return {
                    "ok": False,
                    "status": "degraded",
                    "message": f"Missing tables: {', '.join(missing_tables)}",
                    "tables_found": tables,
                    "tables_required": required_tables
                }
            
            return {
                "ok": True,
                "status": "healthy",
                "database": "connected",
                "tables": tables
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

