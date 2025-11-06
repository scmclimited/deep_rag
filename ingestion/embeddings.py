# embeddings.py
# Unified multi-modal embedding system using CLIP for text and images in same semantic space
import logging
import numpy as np
from typing import Union, List, Optional
from pathlib import Path
from PIL import Image
import io
import os
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lazy loading for CLIP model
_clip_model = None

# Model configuration
# CLIP-ViT-L/14: 768 dimensions (upgraded from ViT-B/32 512 dims)
# Better performance with higher dimensional representations
# Still has 77 token limit for text (inherent to CLIP architecture)
DEFAULT_CLIP_MODEL = os.getenv("CLIP_MODEL", "sentence-transformers/clip-ViT-L-14")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))  # 768 for ViT-L/14, 512 for ViT-B/32

def get_clip_model():
    """Lazy load CLIP model for multi-modal embeddings."""
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

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize embedding vector for cosine similarity."""
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)

def embed_text(text: str, normalize_emb: bool = True, max_length: int = 77) -> np.ndarray:
    """
    Embed text using CLIP model.
    
    CLIP-ViT-B/32 has a maximum sequence length of 77 tokens.
    Longer text will be truncated to fit within this limit using the actual tokenizer.
    
    Args:
        text: Text string to embed
        normalize_emb: Whether to normalize the embedding vector
        max_length: Maximum sequence length (default: 77 for CLIP-ViT-B/32)
        
    Returns:
        Normalized embedding vector (512 dimensions for CLIP-ViT-B/32)
        
    Raises:
        ValueError: If encoding fails even after truncation
    """
    model = get_clip_model()
    
    # Get the actual tokenizer from the CLIP model to do proper token-based truncation
    # For sentence-transformers CLIP models, we need to access the underlying transformers tokenizer
    tokenizer = None
    try:
        # Try multiple ways to access the tokenizer from SentenceTransformer's CLIP module
        if hasattr(model, '_modules') and len(model._modules) > 0:
            first_module = list(model._modules.values())[0]
            
            # Method 1: Direct tokenizer attribute
            if hasattr(first_module, 'tokenizer'):
                tokenizer_obj = first_module.tokenizer
                # CLIPProcessor doesn't have encode, but it might have a tokenizer attribute
                if hasattr(tokenizer_obj, 'encode'):
                    tokenizer = tokenizer_obj
                elif hasattr(tokenizer_obj, 'tokenizer'):
                    # Try nested tokenizer
                    if hasattr(tokenizer_obj.tokenizer, 'encode'):
                        tokenizer = tokenizer_obj.tokenizer
            
            # Method 2: Access from processor if it exists
            if not tokenizer and hasattr(first_module, 'processor'):
                processor = first_module.processor
                if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'encode'):
                    tokenizer = processor.tokenizer
            
            # Method 3: Try to access from the underlying model
            if not tokenizer and hasattr(first_module, '_modules'):
                for module in first_module._modules.values():
                    if hasattr(module, 'tokenizer') and hasattr(module.tokenizer, 'encode'):
                        tokenizer = module.tokenizer
                        break
    except Exception as e:
        logger.debug(f"Could not access tokenizer: {e}, using conservative word-based truncation")
    
    # Truncate using actual tokenizer if available, otherwise use very conservative word-based truncation
    if tokenizer and hasattr(tokenizer, 'encode'):
        try:
            # Tokenize without truncation to check actual length
            # Use add_special_tokens=True to see actual length that will be used
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False, return_tensors=None)
            actual_token_count = len(tokens) if isinstance(tokens, list) else tokens.shape[0] if hasattr(tokens, 'shape') else len(tokens)
            
            if actual_token_count > max_length:
                # Truncate using tokenizer - reserve 2 tokens for special tokens (start/end)
                # Use max_length-2 to ensure we stay under 77 after special tokens are added
                truncate_to = max_length - 2  # Reserve for special tokens
                tokens_truncated = tokenizer.encode(
                    text,
                    add_special_tokens=True,  # Include special tokens in truncation
                    truncation=True,
                    max_length=truncate_to,  # Truncate to max_length-2
                    return_tensors=None
                )
                # Decode back to text (skip special tokens for cleaner output)
                truncated_text = tokenizer.decode(tokens_truncated, skip_special_tokens=True)
                logger.warning(f"Text truncated from {actual_token_count} tokens to ~{truncate_to} tokens using tokenizer (accounting for special tokens)")
                text = truncated_text
        except Exception as e:
            logger.warning(f"Tokenizer truncation failed: {e}, using conservative word-based truncation")
            tokenizer = None
    
    # Fallback: very conservative word-based truncation (20 words max for safety)
    if not tokenizer:
        words = text.split()
        # Very conservative: 20 words ≈ 25-30 tokens (safe margin for CLIP's 77 limit + special tokens)
        max_words = 20
        if len(words) > max_words:
            text = " ".join(words[:max_words])
            logger.warning(f"Text truncated from {len(words)} words to {max_words} words (conservative fallback)")
    
    # Encode with sentence-transformers
    # Try encoding with proper error handling
    max_retries = 3
    for attempt in range(max_retries):
        try:
            emb = model.encode(text, normalize_embeddings=False, show_progress_bar=False, convert_to_numpy=True)
            # Success - exit retry loop
            break
        except Exception as e:
            if attempt < max_retries - 1:
                # More aggressive truncation on retry
                words = text.split()
                # Reduce by 5 words each retry, but ensure we have at least 10 words
                max_words = max(10, len(words) - 5 * (attempt + 1))
                text = " ".join(words[:max_words])
                logger.warning(f"Encoding attempt {attempt + 1} failed: {e}. Retrying with {max_words} words.")
                
                # If we still have tokenizer, try to truncate again with even more conservative limit
                if tokenizer and hasattr(tokenizer, 'encode'):
                    try:
                        # Even more conservative: reserve 4 tokens for special tokens
                        truncate_to = max_length - 4
                        tokens_truncated = tokenizer.encode(
                            text,
                            add_special_tokens=True,
                            truncation=True,
                            max_length=truncate_to,
                            return_tensors=None
                        )
                        text = tokenizer.decode(tokens_truncated, skip_special_tokens=True)
                    except Exception:
                        pass  # Fall back to word-based truncation
            else:
                # Final attempt failed - this is a real error
                logger.error(f"Encoding failed after {max_retries} attempts: {e}")
                raise ValueError(f"Failed to encode text after truncation: {e}. Text may be too long or contain invalid characters.")
    else:
        # This should not happen, but just in case
        raise ValueError("Encoding failed unexpectedly")
    
    if normalize_emb:
        return normalize(emb)
    return emb

def embed_image(image_path: Union[str, Path, Image.Image], normalize_emb: bool = True) -> np.ndarray:
    """
    Embed image using CLIP model.
    
    Args:
        image_path: Path to image file or PIL Image object
        normalize_emb: Whether to normalize the embedding vector
        
    Returns:
        Normalized embedding vector (512 dimensions for CLIP-ViT-B/32)
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
        Normalized embedding vector (512 dimensions)
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
        Array of normalized embedding vectors (N x 512)
    """
    model = get_clip_model()
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

# Embedding dimensions - configured at top of file based on model selection
# CLIP-ViT-L/14: 768 dims (default)
# CLIP-ViT-B/32: 512 dims (legacy)

