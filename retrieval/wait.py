"""
Wait for chunks to be available after ingestion.
"""
import time
import logging
from typing import Optional
from retrieval.db_utils import connect

logger = logging.getLogger(__name__)


def wait_for_chunks(
    doc_id: str, 
    expected_count: Optional[int] = None, 
    max_wait_seconds: int = 40, 
    poll_interval: float = 0.5
) -> int:
    """
    Wait for chunks to be available for a document ID after ingestion.
    
    This ensures that embeddings are complete and chunks are inserted into the database
    before starting query operations.
    
    Args:
        doc_id: Document ID to check
        expected_count: Expected number of chunks (optional, for validation)
        max_wait_seconds: Maximum time to wait in seconds
        poll_interval: Time between checks in seconds
        
    Returns:
        int: Number of chunks found
        
    Raises:
        TimeoutError: If chunks are not available within max_wait_seconds
    """
    start_time = time.time()
    logger.info(f"Waiting for chunks for document {doc_id}...")
    
    while time.time() - start_time < max_wait_seconds:
        try:
            with connect() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM chunks 
                    WHERE doc_id = %s
                """, (doc_id,))
                count = cur.fetchone()[0]
                
                if count > 0:
                    if expected_count is None or count >= expected_count:
                        elapsed = time.time() - start_time
                        logger.info(f"Found {count} chunks for document {doc_id} after {elapsed:.2f} seconds")
                        return count
                    else:
                        logger.debug(f"Found {count} chunks, waiting for {expected_count}...")
                
        except Exception as e:
            logger.warning(f"Error checking chunks for document {doc_id}: {e}")
        
        time.sleep(poll_interval)
    
    # Final check
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) 
                FROM chunks 
                WHERE doc_id = %s
            """, (doc_id,))
            count = cur.fetchone()[0]
            if count > 0:
                logger.warning(f"Found {count} chunks after timeout, proceeding anyway")
                return count
    except Exception as e:
        logger.error(f"Error in final chunk check: {e}")
    
    raise TimeoutError(f"Chunks not available for document {doc_id} after {max_wait_seconds} seconds")

