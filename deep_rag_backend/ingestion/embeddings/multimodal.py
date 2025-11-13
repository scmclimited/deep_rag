"""
Multi-modal embedding utilities.
"""
import logging
import numpy as np
from typing import Union, Optional
from pathlib import Path
from PIL import Image

from ingestion.embeddings.text import embed_text
from ingestion.embeddings.image import embed_image

logger = logging.getLogger(__name__)


def embed_multi_modal(
    text: Optional[str] = None,
    image_path: Optional[Union[str, Path, Image.Image]] = None,
    normalize_emb: bool = True
) -> np.ndarray:
    """
    Embed text and/or image using CLIP model from transformers.
    If both provided, embeds separately and averages them together.
    
    Args:
        text: Optional text string
        image_path: Optional image path or PIL Image
        normalize_emb: Whether to normalize the embedding vector
        
    Returns:
        Normalized embedding vector (768 dimensions for openai/clip-vit-large-patch14-336)
        
    Raises:
        ValueError: If neither text nor image_path is provided
    """
    if text and image_path:
        # Multi-modal: embed text and image separately and combine
        # Use embed_text to handle truncation properly (max 77 tokens)
        text_emb = embed_text(text, normalize_emb=False, max_length=77)
        image_emb = embed_image(image_path, normalize_emb=False)
        
        # Average the embeddings (both are in the same CLIP embedding space)
        combined_emb = (text_emb + image_emb) / 2.0
        
        if normalize_emb:
            from ingestion.embeddings.utils import normalize
            return normalize(combined_emb)
        return combined_emb
    
    elif text:
        return embed_text(text, normalize_emb)
    elif image_path:
        return embed_image(image_path, normalize_emb)
    else:
        raise ValueError("Must provide either text or image_path")

