"""
Batch embedding utilities.
"""
import logging
import numpy as np
from typing import List, Union
from PIL import Image

from ingestion.embeddings.text import embed_text
from ingestion.embeddings.image import embed_image
from ingestion.embeddings.multimodal import embed_multi_modal
from ingestion.embeddings.utils import normalize

logger = logging.getLogger(__name__)


def embed_batch(
    items: List[Union[str, Image.Image, tuple]],
    normalize_emb: bool = True
) -> np.ndarray:
    """
    Embed a batch of text strings, images, or (text, image) tuples.
    
    Args:
        items: List of text strings, PIL Images, or (text, image) tuples
        normalize_emb: Whether to normalize embedding vectors
        
    Returns:
        Array of normalized embedding vectors (N x 768 for CLIP-ViT-L/14)
    """
    embeddings = []
    
    for item in items:
        if isinstance(item, tuple):
            # Multi-modal: (text, image)
            text, image = item
            emb = embed_multi_modal(text=text, image_path=image, normalize_emb=False)
        elif isinstance(item, Image.Image):
            # Image only
            emb = embed_image(item, normalize_emb=False)
        elif isinstance(item, str):
            # Text only
            emb = embed_text(item, normalize_emb=False)
        else:
            raise ValueError(f"Unsupported item type: {type(item)}")
        
        if normalize_emb:
            embeddings.append(normalize(emb))
        else:
            embeddings.append(emb)
    
    return np.array(embeddings)

