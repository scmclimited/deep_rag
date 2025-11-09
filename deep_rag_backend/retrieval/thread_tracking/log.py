"""
Log thread interactions to the database.
"""
import json
import logging
import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, cast

from retrieval.db_utils import connect

logger = logging.getLogger(__name__)


def _json_serializer(value: Any) -> Any:
    """Fallback serializer for objects that aren't JSON serializable by default."""
    try:
        # Handle floats that are NaN/Inf which PostgreSQL JSON does not accept
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, (set, frozenset)):
            iterable_value = cast(Iterable[Any], value)
            return [_json_serializer(v) for v in iterable_value]
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, Decimal):
            return float(value)
        # numpy scalars/arrays expose item()/tolist() helpers
        item_fn = getattr(value, "item", None)
        if callable(item_fn):
            try:
                coerced = item_fn()
                if isinstance(coerced, float) and (math.isnan(coerced) or math.isinf(coerced)):
                    return None
                return coerced
            except Exception:
                pass
        tolist_fn = getattr(value, "tolist", None)
        if callable(tolist_fn):
            try:
                return tolist_fn()
            except Exception:
                pass
    except Exception:
        # Fall-through to final string conversion
        pass
    if value is None:
        return None
    value_obj = cast(object, value)
    value_str: str = str(value_obj)
    return value_str


def _safe_json_dumps(data: Optional[Dict[str, Any]]) -> Optional[str]:
    """Safely serialize arbitrary data structures to JSON."""
    if data is None:
        return None
    try:
        return json.dumps(data, default=_json_serializer)
    except Exception as exc:
        logger.warning("log_thread_interaction: Failed to JSON-encode payload (%s); falling back to string conversion.", exc, exc_info=True)
        try:
            # Final fallback: stringify the payload
            return json.dumps(str(data))
        except Exception:
            return json.dumps(str(data))


def log_thread_interaction(
    user_id: str,
    thread_id: str,
    query_text: Optional[str] = None,
    doc_ids: Optional[List[Any]] = None,
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
        with connect() as conn:
            with conn.cursor() as cur:
                # Ensure doc_ids is stored as TEXT[] (list of strings)
                doc_ids_list: List[str] = []
                if doc_ids:
                    try:
                        doc_ids_list = [str(doc_id) for doc_id in doc_ids if doc_id is not None]
                    except Exception as conversion_error:
                        logger.warning(
                            "log_thread_interaction: Failed to stringify doc_ids %s (error: %s). Falling back to empty list.",
                            doc_ids,
                            conversion_error,
                            exc_info=True,
                        )
                        doc_ids_list = []

                logger.debug(
                    "log_thread_interaction: Inserting interaction user_id=%s thread_id=%s doc_ids=%s",
                    user_id,
                    thread_id,
                    doc_ids_list,
                )

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
                    doc_ids_list,
                    final_answer,
                    _safe_json_dumps(graphstate),
                    _safe_json_dumps(ingestion_meta),
                    entry_point,
                    pipeline_type,
                    cross_doc,
                    _safe_json_dumps(metadata),
                    datetime.now(),
                    datetime.now()
                ))
                result = cur.fetchone()
                if result is None:
                    raise RuntimeError("log_thread_interaction: INSERT did not return an id")
                record_id = int(result[0])
            # Commit outside cursor context to ensure it happens
            conn.commit()
            logger.info(f"Logged thread interaction: user_id={user_id}, thread_id={thread_id}, record_id={record_id}")
            return record_id
    except Exception as e:
        logger.error(f"Failed to log thread interaction: {e}", exc_info=True)
        raise

