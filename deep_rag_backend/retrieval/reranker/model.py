"""
Reranker model loading and configuration.
"""
import os
import logging
from typing import Optional
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)

# Reranker for query time (text-only cross-encoder)
# Default: cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_reranker = None


def get_reranker() -> Optional[CrossEncoder]:
    """
    Get or initialize the reranker model.
    Supports loading from local path (via RERANK_MODEL_PATH env var) or Hugging Face.
    
    Returns:
        CrossEncoder instance, or None if not available
    """
    global _reranker
    if _reranker is None:
        try:
            # Check if we have a local model path
            local_model_path = os.getenv("RERANK_MODEL_PATH")
            
            # If RERANK_MODEL_PATH is not set, check default models directory
            # Models downloaded via download_model.py are stored as: models/{model_name.replace('/', '_')}
            if not local_model_path or not os.path.exists(local_model_path):
                # Try default models directory based on model name
                model_name_safe = RERANK_MODEL.replace('/', '_')
                default_models_path = os.path.join('/app', 'models', model_name_safe)
                if os.path.exists(default_models_path):
                    local_model_path = default_models_path
                    logger.info(f"Found reranker model in default location: {local_model_path}")
            
            # Determine which path to use
            if local_model_path and os.path.exists(local_model_path):
                model_path = local_model_path
                logger.info(f"Loading reranker model from local path: {local_model_path}")
            else:
                model_path = RERANK_MODEL
                logger.info(f"Loading reranker model from Hugging Face: {RERANK_MODEL}")
            
            _reranker = CrossEncoder(model_path)
            logger.info(f"Loaded reranker model: {model_path}")
        except Exception as e:
            logger.warning(f"Reranker not available: {e}. Continuing without reranking.")
            _reranker = None
    return _reranker

