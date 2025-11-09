"""
Log thread interactions to the database.
"""
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from retrieval.db_utils import connect

logger = logging.getLogger(__name__)


def log_thread_interaction(
    user_id: str,
    thread_id: str,
    query_text: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    final_answer: Optional[str] = None,
    graphstate: Optional[Dict[str, Any]] = None,
    ingestion_meta: Optional[Dict[str, Any]] = None,
    entry_point: Optional[str] = None,
    pipeline_type: Optional[str] = None,
    cross_doc: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    """
    Log a thread interaction to the thread_tracking table.
    
    Args:
        user_id: User identifier from external authentication system
        thread_id: Thread/session identifier
        query_text: Original query/question
        doc_ids: List of document IDs retrieved in this thread
        final_answer: Final synthesized answer
        graphstate: Full graph state metadata (all agent steps)
        ingestion_meta: Ingestion metadata (if ingestion occurred)
        entry_point: Entry point used ('cli', 'rest', 'make', 'toml')
        pipeline_type: Pipeline type used ('direct', 'langgraph')
        cross_doc: Whether cross-document retrieval was enabled
        metadata: Additional metadata
        
    Returns:
        int: ID of the inserted record
    """
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO thread_tracking (
                    user_id, thread_id, query_text, doc_ids, final_answer,
                    graphstate, ingestion_meta, entry_point, pipeline_type,
                    cross_doc, metadata, created_at, completed_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """, (
                user_id,
                thread_id,
                query_text,
                doc_ids or [],
                final_answer,
                json.dumps(graphstate) if graphstate else None,
                json.dumps(ingestion_meta) if ingestion_meta else None,
                entry_point,
                pipeline_type,
                cross_doc,
                json.dumps(metadata) if metadata else None,
                datetime.now(),
                datetime.now()
            ))
            record_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Logged thread interaction: user_id={user_id}, thread_id={thread_id}, record_id={record_id}")
            return record_id
    except Exception as e:
        logger.error(f"Failed to log thread interaction: {e}", exc_info=True)
        raise

