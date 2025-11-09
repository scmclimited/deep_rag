"""
Threads route - Manage conversation threads.
"""
import logging
import asyncio
from typing import Optional, Dict
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from retrieval.thread_tracking.get import get_thread_interactions
from retrieval.thread_tracking.log import log_thread_interaction

logger = logging.getLogger(__name__)

router = APIRouter()


class ThreadSeedRequest(BaseModel):
    user_id: str
    thread_id: str


@router.get("/threads")
async def get_threads(
    user_id: Optional[str] = Query(None, description="Filter threads by user ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of threads to return")
) -> Dict:
    """
    Get list of unique threads for a user.
    
    Returns a list of unique thread IDs with their metadata.
    If no threads exist, returns an empty list (not an error).
    """
    try:
        logger.info(f"get_threads: Received request for user_id='{user_id}' (type: {type(user_id)})")
        if not user_id:
            logger.info("get_threads: No user_id provided, returning empty list")
            return {
                "threads": [],
                "count": 0,
                "user_id": None
            }
        # Get all thread interactions for the user
        # Run synchronous DB operation in thread pool to avoid blocking
        logger.info(f"get_threads: Querying database for user_id='{user_id}'")
        interactions = await asyncio.to_thread(get_thread_interactions, user_id=user_id, limit=limit * 10)
        logger.info(f"get_threads: Found {len(interactions)} interactions for user_id='{user_id}'")
        
        # Group by thread_id to get unique threads
        threads_dict = {}
        for interaction in interactions:
            thread_id = interaction.get("thread_id")
            if thread_id and thread_id not in threads_dict:
                threads_dict[thread_id] = {
                    "thread_id": thread_id,
                    "user_id": interaction.get("user_id"),
                    "created_at": interaction.get("created_at"),
                    "last_activity": interaction.get("completed_at") or interaction.get("created_at"),
                    "query_count": 0,
                    "doc_ids": interaction.get("doc_ids") or [],
                    "latest_query": None  # Will be set from most recent interaction
                }
            
            # Update thread metadata
            if thread_id in threads_dict:
                threads_dict[thread_id]["query_count"] += 1
                # Update last activity
                if interaction.get("completed_at"):
                    if not threads_dict[thread_id]["last_activity"] or \
                       interaction.get("completed_at") > threads_dict[thread_id]["last_activity"]:
                        threads_dict[thread_id]["last_activity"] = interaction.get("completed_at")
                        # Update latest query from most recent interaction
                        threads_dict[thread_id]["latest_query"] = interaction.get("query_text")
                # If no completed_at, use created_at for latest query
                elif interaction.get("created_at"):
                    if not threads_dict[thread_id]["last_activity"] or \
                       interaction.get("created_at") > threads_dict[thread_id]["last_activity"]:
                        threads_dict[thread_id]["last_activity"] = interaction.get("created_at")
                        threads_dict[thread_id]["latest_query"] = interaction.get("query_text")
                # Set latest_query if not set yet
                if not threads_dict[thread_id]["latest_query"]:
                    threads_dict[thread_id]["latest_query"] = interaction.get("query_text")
                # Merge doc_ids
                doc_ids = interaction.get("doc_ids") or []
                if doc_ids:
                    existing = set(threads_dict[thread_id]["doc_ids"])
                    threads_dict[thread_id]["doc_ids"] = list(existing | set(doc_ids))
        
        # Convert to list and sort by last activity
        threads_list = list(threads_dict.values())
        threads_list.sort(key=lambda x: x.get("last_activity") or "", reverse=True)
        
        # Limit results
        threads_list = threads_list[:limit]
        
        return {
            "threads": threads_list,
            "count": len(threads_list),
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error getting threads: {e}", exc_info=True)
        # Return empty list instead of error if no threads exist
        return {
            "threads": [],
            "count": 0,
            "user_id": user_id
        }


@router.post("/threads")
async def seed_thread(body: ThreadSeedRequest) -> Dict:
    """
    Seed a new thread for a user so it appears immediately before interactions are logged.
    """
    try:
        logger.info("seed_thread: Seeding thread_id='%s' for user_id='%s'", body.thread_id, body.user_id)

        existing = await asyncio.to_thread(
            get_thread_interactions,
            user_id=body.user_id,
            thread_id=body.thread_id,
            limit=1,
        )
        if existing:
            logger.info("seed_thread: Thread already exists, skipping seed")
            return {
                "status": "exists",
                "thread_id": body.thread_id,
                "user_id": body.user_id,
            }

        record_id = await asyncio.to_thread(
            log_thread_interaction,
            user_id=body.user_id,
            thread_id=body.thread_id,
            query_text=None,
            final_answer=None,
            graphstate=None,
            ingestion_meta=None,
            entry_point="rest",
            pipeline_type="seed",
            cross_doc=False,
            metadata={"seed": True},
        )

        logger.info(
            "seed_thread: Seeded thread thread_id='%s' user_id='%s' record_id=%s",
            body.thread_id,
            body.user_id,
            record_id,
        )

        return {
            "status": "seeded",
            "thread_id": body.thread_id,
            "user_id": body.user_id,
            "record_id": record_id,
        }
    except Exception as exc:
        logger.error("seed_thread: Failed to seed thread: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/threads/debug/user-ids")
def get_all_user_ids() -> Dict:
    """
    Debug endpoint to get all unique user_ids in the database.
    This helps diagnose why threads aren't being found for a specific user_id.
    """
    try:
        from retrieval.thread_tracking.get import get_thread_interactions
        from retrieval.db_utils import connect
        
        with connect() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT user_id, COUNT(*) as thread_count
                FROM thread_tracking
                GROUP BY user_id
                ORDER BY thread_count DESC
            """)
            rows = cur.fetchall()
            user_ids = [{"user_id": row[0], "thread_count": row[1]} for row in rows]
            
            return {
                "user_ids": user_ids,
                "total_unique_users": len(user_ids),
                "message": "This is a debug endpoint to help diagnose thread loading issues"
            }
    except Exception as e:
        logger.error(f"Error getting all user_ids: {e}", exc_info=True)
        return {
            "error": str(e),
            "user_ids": [],
            "total_unique_users": 0
        }


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str, user_id: Optional[str] = Query(None)) -> Dict:
    """
    Get thread details and message history.
    Converts interactions to messages format for frontend compatibility.
    """
    try:
        # Run synchronous DB operation in thread pool to avoid blocking
        if user_id:
            interactions = await asyncio.to_thread(get_thread_interactions, thread_id=thread_id, user_id=user_id, limit=100)
        else:
            interactions = await asyncio.to_thread(get_thread_interactions, thread_id=thread_id, limit=100)
        
        if not interactions:
            raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
        
        # Filter by user_id if provided
        if user_id:
            interactions = [interaction for interaction in interactions if interaction.get("user_id") == user_id]

        if not interactions:
            raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

        # Get thread metadata from first interaction
        first_interaction = interactions[0]
        
        # Convert interactions to messages format
        messages = []
        for interaction in interactions:
            # Extract data from interaction
            query_text = interaction.get("query_text")
            final_answer = interaction.get("final_answer")
            doc_ids = interaction.get("doc_ids") or []
            created_at = interaction.get("created_at")
            graphstate = interaction.get("graphstate") or {}
            
            # Extract additional info from graphstate if available
            doc_id = graphstate.get("doc_id") or (doc_ids[0] if doc_ids else None)
            pages = graphstate.get("pages") or []
            confidence = graphstate.get("confidence", 0.0)
            action = graphstate.get("action", "answer")
            
            # Add user message
            if query_text:
                messages.append({
                    "role": "user",
                    "content": query_text,
                    "timestamp": created_at or "",
                    "attachment": None
                })
            
            # Add assistant message
            if final_answer:
                messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "timestamp": interaction.get("completed_at") or created_at or "",
                    "doc_id": doc_id,
                    "doc_ids": doc_ids,
                    "doc_title": None,  # Could be fetched if needed
                    "pages": pages,
                    "confidence": confidence,
                    "action": action
                })
        
        thread_data = {
            "thread_id": thread_id,
            "user_id": first_interaction.get("user_id"),
            "created_at": first_interaction.get("created_at"),
            "messages": messages,
            "message_count": len(messages),
            "interactions": interactions  # Keep for backward compatibility
        }
        
        return thread_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

