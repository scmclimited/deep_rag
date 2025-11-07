"""
Image embedding utilities.
"""
import logging
import numpy as np
from typing import Union
from pathlib import Path
from PIL import Image

from ingestion.embeddings.model import get_clip_model
from ingestion.embeddings.utils import normalize

logger = logging.getLogger(__name__)


def embed_image(image_path: Union[str, Path, Image.Image], normalize_emb: bool = True) -> np.ndarray:
    """
    Embed image using CLIP model.
    
    Args:
        image_path: Path to image file or PIL Image object
        normalize_emb: Whether to normalize the embedding vector
        
    Returns:
        Normalized embedding vector (768 dimensions for CLIP-ViT-L/14)
    """
    model = get_clip_model()
    
    # Handle different input types
    if isinstance(image_path, (str, Path)):
        image = Image.open(image_path).convert('RGB')
    elif isinstance(image_path, Image.Image):
        image = image_path.convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image_path)}")
    
    # CLIP's encode method handles images automatically
    emb = model.encode(image, normalize_embeddings=False)
    if normalize_emb:
        return normalize(emb)
    return emb

