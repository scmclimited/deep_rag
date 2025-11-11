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

# Minimum image size for CLIP (ViT models typically expect at least 32x32)
MIN_IMAGE_SIZE = 32


def _validate_and_resize_image(image: Image.Image) -> Image.Image:
    """
    Validate and resize image if it's too small for CLIP processing.
    
    Args:
        image: PIL Image object
        
    Returns:
        Validated/resized PIL Image object
    """
    width, height = image.size
    
    # Check if image is too small (less than minimum size in either dimension)
    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
        logger.warning(
            f"Image size ({width}x{height}) is below minimum ({MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}). "
            f"Resizing to minimum size while maintaining aspect ratio."
        )
        
        # Calculate new size maintaining aspect ratio
        if width < height:
            new_width = MIN_IMAGE_SIZE
            new_height = max(MIN_IMAGE_SIZE, int(height * (MIN_IMAGE_SIZE / width)))
        else:
            new_height = MIN_IMAGE_SIZE
            new_width = max(MIN_IMAGE_SIZE, int(width * (MIN_IMAGE_SIZE / height)))
        
        # Use LANCZOS resampling for better quality when upscaling
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.debug(f"Resized image to {new_width}x{new_height}")
    
    return image


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
    
    # Validate and resize if necessary
    image = _validate_and_resize_image(image)
    
    # CLIP's encode method handles images automatically
    emb = model.encode(image, normalize_embeddings=False)
    if normalize_emb:
        return normalize(emb)
    return emb

