"""
CLIP model loading and configuration.
"""
import os
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lazy loading for CLIP model
_clip_model = None

# Model configuration
# CLIP-ViT-L/14: 768 dimensions (upgraded from ViT-B/32 512 dims)
# Better performance with higher dimensional representations
# Still has 77 token limit for text (inherent to CLIP architecture)
DEFAULT_CLIP_MODEL = os.getenv("CLIP_MODEL")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM"))  # 768 for ViT-L/14, 512 for ViT-B/32


def get_clip_model() -> SentenceTransformer:
    """
    Lazy load CLIP model for multi-modal embeddings.
    
    Returns:
        SentenceTransformer CLIP model instance
        
    Raises:
        ImportError: If CLIP model cannot be loaded
    """
    global _clip_model
    if _clip_model is None:
        try:
            # CLIP model that handles both text and images in same embedding space
            # Default: ViT-L/14 (768 dims) for better quality
            # Alternative: ViT-B/32 (512 dims) for faster performance
            # Can be configured via CLIP_MODEL environment variable
            _clip_model = SentenceTransformer(DEFAULT_CLIP_MODEL)
            logger.info(f"Loaded CLIP multi-modal embedding model ({DEFAULT_CLIP_MODEL}, {EMBEDDING_DIM} dims)")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise ImportError(
                f"CLIP model not available: {e}\n"
                "Install with: pip install sentence-transformers\n"
                f"Failed model: {DEFAULT_CLIP_MODEL}"
            ) from e
    return _clip_model

