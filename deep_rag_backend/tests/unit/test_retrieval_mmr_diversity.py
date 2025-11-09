"""
Unit tests for retrieval.mmr module - Diversity functionality.
"""
import pytest
import numpy as np
from retrieval.mmr import mmr


def test_mmr_diversity():
    """Test that MMR promotes diversity."""
    query_emb = np.random.rand(768).astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)
    
    # Create candidates where some are very similar
    base_emb = np.random.rand(768).astype(np.float32)
    base_emb = base_emb / np.linalg.norm(base_emb)
    
    candidates = [
        {"chunk_id": "1", "emb": base_emb, "text": "chunk 1"},
        {"chunk_id": "2", "emb": base_emb + 0.01 * np.random.rand(768).astype(np.float32), "text": "chunk 2"},
        {"chunk_id": "3", "emb": np.random.rand(768).astype(np.float32), "text": "chunk 3"}
    ]
    
    # Normalize
    for c in candidates:
        c["emb"] = c["emb"] / np.linalg.norm(c["emb"])
    
    result = mmr(candidates, query_emb, lambda_mult=0.5, k=2)
    
    # Should select diverse chunks
    assert len(result) == 2
    assert result[0]["chunk_id"] != result[1]["chunk_id"]

