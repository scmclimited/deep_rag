"""
Reranking functions for query-time reranking.
"""
import logging
from typing import List, Dict
from retrieval.reranker.model import get_reranker

logger = logging.getLogger(__name__)


def rerank_candidates(query: str, candidates: List[Dict]) -> List[Dict]:
    """
    Rerank candidates using cross-encoder.
    
    Args:
        query: Query string
        candidates: List of candidate chunks with 'text' key
        
    Returns:
        List of candidates with 'ce' (cross-encoder) score added, sorted by score
    """
    reranker = get_reranker()
    if not reranker or not candidates:
        return candidates
    
    try:
        pairs = [[query, c["text"]] for c in candidates]
        ce_scores = reranker.predict(pairs)
        for c, s in zip(candidates, ce_scores):
            c["ce"] = float(s)
        candidates.sort(key=lambda x: x.get("ce", 0.0), reverse=True)
    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Continuing without reranking.")
    
    return candidates

