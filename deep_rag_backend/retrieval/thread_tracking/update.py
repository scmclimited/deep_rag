"""
Update thread interactions in the database.
"""
import json
import logging
from typing import Optional, List, Dict, Any, Union
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


def archive_thread(thread_id: str, user_id: str, archived: bool = True) -> bool:
    """
    Archive or unarchive a thread by updating all its interactions' metadata.
    
    Args:
        thread_id: Thread ID to archive/unarchive
        user_id: User ID (for security - only archive user's own threads)
        archived: True to archive, False to unarchive
        
    Returns:
        bool: True if update was successful
    """
    try:
        with connect() as conn, conn.cursor() as cur:
            # First, get existing metadata for all interactions in this thread
            cur.execute("""
                SELECT id, metadata
                FROM thread_tracking
                WHERE thread_id = %s AND user_id = %s
            """, (thread_id, user_id))
            
            rows = cur.fetchall()
            if not rows:
                logger.warning(f"No threads found to archive: thread_id={thread_id}, user_id={user_id}")
                return False
            
            # Update each interaction's metadata
            updated_count = 0
            for row in rows:
                record_id = row[0]
                existing_metadata = _safe_json_load("metadata", row[1]) or {}
                
                # Update archived status
                existing_metadata["archived"] = archived
                if archived:
                    existing_metadata["archived_at"] = datetime.now().isoformat()
                else:
                    existing_metadata.pop("archived_at", None)
                
                # Update the record
                cur.execute("""
                    UPDATE thread_tracking
                    SET metadata = %s
                    WHERE id = %s
                """, (json.dumps(existing_metadata), record_id))
                updated_count += 1
            
            conn.commit()
            logger.info(f"Archived/unarchived thread: thread_id={thread_id}, user_id={user_id}, archived={archived}, updated {updated_count} records")
            return True
    except Exception as e:
        logger.error(f"Failed to archive thread: {e}", exc_info=True)
        return False


def _safe_json_load(field_name: str, value: Union[str, bytes, bytearray, Dict[str, Any], List[Any], None]) -> Optional[Any]:
    """
    Safely load JSON data that might already be decoded or stored as bytes.
    """
    if value is None:
        return None
    
    if isinstance(value, (dict, list)):
        return value
    
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return stripped
    
    return value
