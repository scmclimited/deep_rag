"""
Embeddings package - Unified multi-modal embedding system using CLIP.
"""
from ingestion.embeddings.model import get_clip_model, DEFAULT_CLIP_MODEL, EMBEDDING_DIM
from ingestion.embeddings.utils import normalize
from ingestion.embeddings.text import embed_text
from ingestion.embeddings.image import embed_image
from ingestion.embeddings.multimodal import embed_multi_modal
from ingestion.embeddings.batch import embed_batch

__all__ = [
    "get_clip_model",
    "DEFAULT_CLIP_MODEL",
    "EMBEDDING_DIM",
    "normalize",
    "embed_text",
    "embed_image",
    "embed_multi_modal",
    "embed_batch",
]

