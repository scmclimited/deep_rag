"""
Reranker model loading and configuration.
"""
import logging
from typing import Optional
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Reranker for query time (text-only cross-encoder)
RERANK_MODEL = "BAAI/bge-reranker-base"
_reranker = None


def get_reranker() -> Optional[CrossEncoder]:
    """
    Get or initialize the reranker model.
    
    Returns:
        CrossEncoder instance, or None if not available
    """
    global _reranker
    if _reranker is None:
        try:
            _reranker = CrossEncoder(RERANK_MODEL)
            logger.info(f"Loaded reranker model: {RERANK_MODEL}")
        except Exception as e:
            logger.warning(f"Reranker not available: {e}. Continuing without reranking.")
            _reranker = None
    return _reranker

