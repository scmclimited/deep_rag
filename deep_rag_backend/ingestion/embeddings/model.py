"""
CLIP model loading and configuration.
"""
import os
import logging
import torch
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

# Model configuration
# openai/clip-vit-large-patch14-336: 768 dimensions
# Text encoder: 77 token limit (inherent to CLIP architecture)
# Image encoder: 336x336 pixels input size
DEFAULT_CLIP_MODEL = os.getenv("CLIP_MODEL")
EMBEDDING_DIM_ENV = os.getenv("EMBEDDING_DIM")

# Validate required environment variables
if DEFAULT_CLIP_MODEL is None:
    raise ValueError(
        "CLIP_MODEL environment variable is not set. "
        "Please set CLIP_MODEL in your .env file. "
        "Example: CLIP_MODEL=openai/clip-vit-large-patch14-336"
    )

if EMBEDDING_DIM_ENV is None:
    raise ValueError(
        "EMBEDDING_DIM environment variable is not set. "
        "Please set EMBEDDING_DIM in your .env file. "
        "Example: EMBEDDING_DIM=768 (for openai/clip-vit-large-patch14-336)"
    )

try:
    EMBEDDING_DIM = int(EMBEDDING_DIM_ENV)
except (ValueError, TypeError) as e:
    raise ValueError(
        f"EMBEDDING_DIM must be a valid integer. Got: {EMBEDDING_DIM_ENV}"
    ) from e

# Global variables for model and processor
_clip_model = None
_clip_processor = None


def get_clip_model() -> CLIPModel:
    """
    Load CLIP model for multi-modal embeddings.
    Supports loading from local path (via CLIP_MODEL_PATH env var) or Hugging Face.
    
    Returns:
        CLIPModel instance from transformers
        
    Raises:
        ImportError: If CLIP model cannot be loaded
    """
    global _clip_model, _clip_processor
    
    if _clip_model is None:
        try:
            # Check if we have a local model path
            local_model_path = os.getenv("CLIP_MODEL_PATH")
            
            # If CLIP_MODEL_PATH is not set, check default models directory
            # Models downloaded via download_model.py are stored as: models/{model_name.replace('/', '_')}
            if not local_model_path or not os.path.exists(local_model_path):
                # Try default models directory based on model name
                model_name_safe = DEFAULT_CLIP_MODEL.replace('/', '_')
                default_models_path = os.path.join('/app', 'models', model_name_safe)
                if os.path.exists(default_models_path):
                    local_model_path = default_models_path
                    logger.info(f"Found model in default location: {local_model_path}")
            
            # Determine which path to use
            if local_model_path and os.path.exists(local_model_path):
                model_path = local_model_path
                logger.info(f"Loading CLIP model from local path: {local_model_path}")
            else:
                model_path = DEFAULT_CLIP_MODEL
                logger.info(f"Loading CLIP model from Hugging Face: {DEFAULT_CLIP_MODEL}")
            
            _clip_model = CLIPModel.from_pretrained(model_path)
            _clip_processor = CLIPProcessor.from_pretrained(model_path)
            
            # Set model to evaluation mode
            _clip_model.eval()
            
            # Validate that the model is properly initialized
            if _clip_model is None:
                raise ValueError(
                    f"CLIP model '{DEFAULT_CLIP_MODEL}' failed to load. "
                    "Check that the model name is correct and the model files are available."
                )
            
            # Test that the model can actually encode (functional validation)
            try:
                # Test text encoding
                test_text = "test"
                inputs = _clip_processor(text=[test_text], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    text_features = _clip_model.get_text_features(**inputs)
                    test_embedding = text_features[0].cpu().numpy()
                
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
            _clip_model = None
            _clip_processor = None
            raise ImportError(
                f"CLIP model not available: {e}\n"
                "Install with: pip install transformers torch\n"
                f"Failed model: {DEFAULT_CLIP_MODEL}\n"
                "Make sure CLIP_MODEL is set correctly in your .env file."
            ) from e
    
    return _clip_model


def get_clip_processor() -> CLIPProcessor:
    """
    Get CLIP processor for tokenization and image preprocessing.
    
    Returns:
        CLIPProcessor instance from transformers
        
    Raises:
        ImportError: If CLIP processor cannot be loaded
    """
    global _clip_processor
    
    # Ensure model is loaded (which also loads processor)
    if _clip_processor is None:
        get_clip_model()  # This will load both model and processor
    
    return _clip_processor

