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
EMBEDDING_DIM_ENV = os.getenv("EMBEDDING_DIM")

# Validate required environment variables
if DEFAULT_CLIP_MODEL is None:
    raise ValueError(
        "CLIP_MODEL environment variable is not set. "
        "Please set CLIP_MODEL in your .env file. "
        "Example: CLIP_MODEL=sentence-transformers/clip-ViT-L-14"
    )

if EMBEDDING_DIM_ENV is None:
    raise ValueError(
        "EMBEDDING_DIM environment variable is not set. "
        "Please set EMBEDDING_DIM in your .env file. "
        "Example: EMBEDDING_DIM=768 (for CLIP-ViT-L/14) or EMBEDDING_DIM=512 (for CLIP-ViT-B/32)"
    )

try:
    EMBEDDING_DIM = int(EMBEDDING_DIM_ENV)
except (ValueError, TypeError) as e:
    raise ValueError(
        f"EMBEDDING_DIM must be a valid integer. Got: {EMBEDDING_DIM_ENV}"
    ) from e


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
            
            # Validate that the model is properly initialized
            # Check that _first_module() is not None (indicates model loaded correctly)
            try:
                first_module = _clip_model._first_module()
                if first_module is None:
                    raise ValueError(
                        f"CLIP model '{DEFAULT_CLIP_MODEL}' loaded but has no modules. "
                        "This usually means the model failed to load properly. "
                        "Check that the model name is correct and the model files are available."
                    )
            except AttributeError:
                # If _first_module() doesn't exist, check _modules directly
                if not hasattr(_clip_model, '_modules') or len(_clip_model._modules) == 0:
                    raise ValueError(
                        f"CLIP model '{DEFAULT_CLIP_MODEL}' loaded but has no modules. "
                        "This usually means the model failed to load properly. "
                        "Check that the model name is correct and the model files are available."
                    )
                # Check that first module is not None
                first_module = list(_clip_model._modules.values())[0] if _clip_model._modules else None
                if first_module is None:
                    raise ValueError(
                        f"CLIP model '{DEFAULT_CLIP_MODEL}' loaded but first module is None. "
                        "This usually means the model failed to load properly. "
                        "Check that the model name is correct and the model files are available."
                    )
            
            # Test that the model can actually encode (functional validation)
            try:
                test_embedding = _clip_model.encode("test", convert_to_numpy=True)
                if test_embedding is None or len(test_embedding) == 0:
                    raise ValueError("Model encoding test returned empty result")
                if len(test_embedding) != EMBEDDING_DIM:
                    raise ValueError(
                        f"Model embedding dimension mismatch: expected {EMBEDDING_DIM}, "
                        f"got {len(test_embedding)}. Check EMBEDDING_DIM matches the model."
                    )
            except Exception as e:
                raise ValueError(
                    f"CLIP model '{DEFAULT_CLIP_MODEL}' loaded but encoding test failed: {e}. "
                    "The model may not be properly initialized or compatible."
                ) from e
            
            logger.info(f"Loaded CLIP multi-modal embedding model ({DEFAULT_CLIP_MODEL}, {EMBEDDING_DIM} dims)")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            _clip_model = None  # Reset on failure
            raise ImportError(
                f"CLIP model not available: {e}\n"
                "Install with: pip install sentence-transformers\n"
                f"Failed model: {DEFAULT_CLIP_MODEL}\n"
                "Make sure CLIP_MODEL is set correctly in your .env file."
            ) from e
    return _clip_model

