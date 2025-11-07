"""
Merge and deduplicate chunks from two retrieval stages.
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def merge_and_deduplicate(primary_chunks: List[Dict], secondary_chunks: List[Dict], k: int) -> List[Dict]:
    """
    Merge and deduplicate chunks from two retrieval stages, prioritizing primary chunks.
    
    Args:
        primary_chunks: Chunks from primary retrieval stage
        secondary_chunks: Chunks from secondary retrieval stage
        k: Maximum number of chunks to return
        
    Returns:
        Merged and deduplicated list of chunks
    """
    seen_chunk_ids = set()
    merged = []
    
    # First, add all primary chunks (prioritized)
    for chunk in primary_chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            merged.append(chunk)
    
    # Then, add secondary chunks that aren't duplicates
    for chunk in secondary_chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            merged.append(chunk)
    
    # Sort by combined score (prioritize primary chunks with higher scores)
    # Primary chunks get a boost in scoring
    for i, chunk in enumerate(merged):
        if i < len(primary_chunks):
            # Boost primary chunks
            chunk["_priority_score"] = chunk.get("ce", chunk.get("vec", 0.0)) + 0.1
        else:
            chunk["_priority_score"] = chunk.get("ce", chunk.get("vec", 0.0))
    
    merged.sort(key=lambda x: x.get("_priority_score", 0.0), reverse=True)
    
    # Remove temporary priority score
    for chunk in merged:
        chunk.pop("_priority_score", None)
    
    logger.info(f"Merged {len(primary_chunks)} primary chunks with {len(secondary_chunks)} secondary chunks, "
                f"result: {len(merged)} unique chunks (returning top {k})")
    
    return merged[:k]

