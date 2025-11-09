"""
Multi-modal embedding utilities.
"""
import logging
import numpy as np
from typing import Union, Optional
from pathlib import Path
from PIL import Image

from ingestion.embeddings.model import get_clip_model
from ingestion.embeddings.text import embed_text
from ingestion.embeddings.image import embed_image
from ingestion.embeddings.utils import normalize

logger = logging.getLogger(__name__)


def embed_multi_modal(
    text: Optional[str] = None,
    image_path: Optional[Union[str, Path, Image.Image]] = None,
    normalize_emb: bool = True
) -> np.ndarray:
    """
    Embed text and/or image using CLIP model.
    If both provided, concatenates and embeds together.
    
    Args:
        text: Optional text string
        image_path: Optional image path or PIL Image
        normalize_emb: Whether to normalize the embedding vector
        
    Returns:
        Normalized embedding vector (768 dimensions for CLIP-ViT-L/14)
        
    Raises:
        ValueError: If neither text nor image_path is provided
    """
    model = get_clip_model()
    
    if text and image_path:
        # Multi-modal: embed text and image together
        # CLIP can handle both, but we'll embed separately and combine
        # Use embed_text to handle truncation properly (max 77 tokens)
        text_emb = embed_text(text, normalize_emb=False, max_length=77)
        
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image_path)}")
        
        image_emb = model.encode(image, normalize_embeddings=False)
        
        # Average the embeddings (alternatively, could concatenate and reduce)
        combined_emb = (text_emb + image_emb) / 2.0
        
        if normalize_emb:
            return normalize(combined_emb)
        return combined_emb
    
    elif text:
        return embed_text(text, normalize_emb)
    elif image_path:
        return embed_image(image_path, normalize_emb)
    else:
        raise ValueError("Must provide either text or image_path")

