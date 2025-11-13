"""
Text embedding utilities.
"""
import logging
import numpy as np
import torch
from ingestion.embeddings.model import get_clip_model, get_clip_processor
from ingestion.embeddings.utils import normalize

logger = logging.getLogger(__name__)


def embed_text(text: str, normalize_emb: bool = True, max_length: int = 77) -> np.ndarray:
    """
    Embed text using CLIP model from transformers.
    
    CLIP has a maximum sequence length of 77 tokens.
    Longer text will be truncated to fit within this limit using the processor.
    
    Args:
        text: Text string to embed
        normalize_emb: Whether to normalize the embedding vector
        max_length: Maximum sequence length (default: 77 for CLIP)
        
    Returns:
        Normalized embedding vector (768 dimensions for openai/clip-vit-large-patch14-336)
        
    Raises:
        ValueError: If encoding fails even after truncation
    """
    model = get_clip_model()
    processor = get_clip_processor()
    
    # Process text with CLIP processor (handles tokenization and truncation)
    try:
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Get text features from the model
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            emb = text_features[0].cpu().numpy()
        
        if emb is None or len(emb) == 0:
            raise ValueError("Model encoding returned empty result")
        
    except Exception as e:
        logger.error(f"Failed to encode text: {e}")
        # Try with more aggressive truncation
        try:
            # Fallback: word-based truncation
            words = text.split()
            max_words = 20  # Conservative: 20 words ≈ 25-30 tokens
            if len(words) > max_words:
                text = " ".join(words[:max_words])
                logger.warning(f"Text truncated from {len(words)} words to {max_words} words")
            
            inputs = processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)
                emb = text_features[0].cpu().numpy()
            
            if emb is None or len(emb) == 0:
                raise ValueError("Model encoding returned empty result after truncation")
        except Exception as e2:
            raise ValueError(f"Failed to encode text after truncation: {e2}. Text may be too long or contain invalid characters.") from e2
    
    if normalize_emb:
        return normalize(emb)
    return emb

