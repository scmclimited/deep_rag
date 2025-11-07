"""
Unit tests for retrieval.mmr module - Basic functionality.
"""
import pytest
import numpy as np
from retrieval.mmr import mmr


def test_mmr_basic():
    """Test basic MMR functionality."""
    query_emb = np.random.rand(768).astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)  # Normalize
    
    candidates = [
        {
            "chunk_id": "1",
            "emb": np.random.rand(768).astype(np.float32),
            "text": "chunk 1"
        },
        {
            "chunk_id": "2",
            "emb": np.random.rand(768).astype(np.float32),
            "text": "chunk 2"
        },
        {
            "chunk_id": "3",
            "emb": np.random.rand(768).astype(np.float32),
            "text": "chunk 3"
        }
    ]
    
    # Normalize candidate embeddings
    for c in candidates:
        c["emb"] = c["emb"] / np.linalg.norm(c["emb"])
    
    result = mmr(candidates, query_emb, lambda_mult=0.5, k=2)
    
    assert len(result) == 2
    assert all("chunk_id" in r for r in result)
    assert all("emb" in r for r in result)

