"""
Embedding utility functions.
"""
import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize embedding vector for cosine similarity.
    
    Args:
        v: Embedding vector
        
    Returns:
        Normalized embedding vector
    """
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)

