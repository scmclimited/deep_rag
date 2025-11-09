"""
Retrieve thread interactions from the database.
"""
import json
import logging
from typing import Optional, List, Dict, Any
from retrieval.db_utils import connect

logger = logging.getLogger(__name__)


def get_thread_interactions(
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Retrieve thread interactions from the database.
    
    Args:
        user_id: Filter by user_id (optional)
        thread_id: Filter by thread_id (optional)
        limit: Maximum number of records to return
        
    Returns:
        List of thread interaction records
    """
    try:
        logger.info(f"get_thread_interactions: Querying with user_id='{user_id}', thread_id='{thread_id}', limit={limit}")
        with connect() as conn, conn.cursor() as cur:
            conditions = []
            params = []
            
            if user_id:
                conditions.append("user_id = %s")
                params.append(user_id)
                logger.info(f"get_thread_interactions: Added condition user_id='{user_id}'")
            
            if thread_id:
                conditions.append("thread_id = %s")
                params.append(thread_id)
                logger.info(f"get_thread_interactions: Added condition thread_id='{thread_id}'")
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)
            
            query = f"""
                SELECT id, user_id, thread_id, query_text, doc_ids, final_answer,
                       graphstate, ingestion_meta, entry_point, pipeline_type,
                       cross_doc, metadata, created_at, completed_at
                FROM thread_tracking
                {where_clause}
                ORDER BY created_at DESC
                LIMIT %s
            """
            logger.info(f"get_thread_interactions: Executing query: {query.strip()}")
            logger.info(f"get_thread_interactions: With params: {params}")
            
            cur.execute(query, params)
            
            rows = cur.fetchall()
            logger.info(f"get_thread_interactions: Query returned {len(rows)} rows")
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "user_id": row[1],
                    "thread_id": row[2],
                    "query_text": row[3],
                    "doc_ids": row[4],
                    "final_answer": row[5],
                    "graphstate": json.loads(row[6]) if row[6] else None,
                    "ingestion_meta": json.loads(row[7]) if row[7] else None,
                    "entry_point": row[8],
                    "pipeline_type": row[9],
                    "cross_doc": row[10],
                    "metadata": json.loads(row[11]) if row[11] else None,
                    "created_at": row[12].isoformat() if row[12] else None,
                    "completed_at": row[13].isoformat() if row[13] else None
                })
            return results
    except Exception as e:
        logger.error(f"Failed to retrieve thread interactions: {e}", exc_info=True)
        return []

