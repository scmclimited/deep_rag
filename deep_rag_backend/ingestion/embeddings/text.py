"""
Text embedding utilities.
"""
import logging
import numpy as np
from ingestion.embeddings.model import get_clip_model
from ingestion.embeddings.utils import normalize

logger = logging.getLogger(__name__)


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
        Normalized embedding vector (768 dimensions for CLIP-ViT-L/14)
        
    Raises:
        ValueError: If encoding fails even after truncation
    """
    model = get_clip_model()
    
    # Get the actual tokenizer from the CLIP model to do proper token-based truncation
    # For sentence-transformers CLIP models, we need to access the underlying transformers tokenizer
    tokenizer = None
    try:
        # Ensure model is properly initialized
        if model is None:
            logger.warning("CLIP model is None, cannot access tokenizer")
        # Try multiple ways to access the tokenizer from SentenceTransformer's CLIP module
        elif hasattr(model, '_modules') and len(model._modules) > 0:
            first_module = list(model._modules.values())[0]
            
            # Ensure first_module is not None before accessing attributes
            if first_module is None:
                logger.warning("First module of CLIP model is None, cannot access tokenizer")
            # Method 1: Direct tokenizer attribute
            elif hasattr(first_module, 'tokenizer'):
                tokenizer_obj = first_module.tokenizer
                # CLIPProcessor doesn't have encode, but it might have a tokenizer attribute
                if hasattr(tokenizer_obj, 'encode'):
                    tokenizer = tokenizer_obj
                elif hasattr(tokenizer_obj, 'tokenizer'):
                    # Try nested tokenizer
                    if hasattr(tokenizer_obj.tokenizer, 'encode'):
                        tokenizer = tokenizer_obj.tokenizer
            
            # Method 2: Access from processor if it exists
            if first_module is not None and not tokenizer and hasattr(first_module, 'processor'):
                processor = first_module.processor
                if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'encode'):
                    tokenizer = processor.tokenizer
            
            # Method 3: Try to access from the underlying model
            if first_module is not None and not tokenizer and hasattr(first_module, '_modules'):
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
    emb = None
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
    
    if emb is None:
        # This should not happen, but just in case
        raise ValueError("Encoding failed unexpectedly")
    
    if normalize_emb:
        return normalize(emb)
    return emb

