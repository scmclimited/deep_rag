"""
Update thread interactions in the database.
"""
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from retrieval.db_utils import connect

logger = logging.getLogger(__name__)


def update_thread_interaction(
    record_id: int,
    final_answer: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    graphstate: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Update an existing thread interaction record.
    
    Args:
        record_id: ID of the record to update
        final_answer: Final synthesized answer
        doc_ids: Updated list of document IDs
        graphstate: Updated graph state metadata
        metadata: Updated additional metadata
        
    Returns:
        bool: True if update was successful
    """
    try:
        updates = []
        params = []
        
        if final_answer is not None:
            updates.append("final_answer = %s")
            params.append(final_answer)
        
        if doc_ids is not None:
            updates.append("doc_ids = %s")
            params.append(doc_ids)
        
        if graphstate is not None:
            updates.append("graphstate = %s")
            params.append(json.dumps(graphstate))
        
        if metadata is not None:
            updates.append("metadata = %s")
            params.append(json.dumps(metadata))
        
        if updates:
            updates.append("completed_at = %s")
            params.append(datetime.now())
            params.append(record_id)
            
            with connect() as conn, conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE thread_tracking
                    SET {', '.join(updates)}
                    WHERE id = %s
                """, params)
                conn.commit()
                logger.info(f"Updated thread interaction: record_id={record_id}")
                return True
        return False
    except Exception as e:
        logger.error(f"Failed to update thread interaction: {e}", exc_info=True)
        return False

