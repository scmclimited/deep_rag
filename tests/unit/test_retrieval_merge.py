"""
Unit tests for retrieval chunk merging and deduplication.
"""
import pytest
from retrieval.retrieval import _merge_and_deduplicate


class TestMergeAndDeduplicate:
    """Tests for chunk merging and deduplication."""
    
    def test_merge_primary_chunks(self):
        """Test merging with primary chunks only."""
        primary = [
            {"chunk_id": "1", "text": "test1", "vec": 0.9},
            {"chunk_id": "2", "text": "test2", "vec": 0.8}
        ]
        secondary = []
        result = _merge_and_deduplicate(primary, secondary, k=2)
        assert len(result) == 2
        assert result[0]["chunk_id"] == "1"
    
    def test_merge_with_duplicates(self):
        """Test merging with duplicate chunks."""
        primary = [{"chunk_id": "1", "text": "test1", "vec": 0.9}]
        secondary = [{"chunk_id": "1", "text": "test1", "vec": 0.9}]
        result = _merge_and_deduplicate(primary, secondary, k=2)
        assert len(result) == 1  # Duplicate removed
    
    def test_merge_prioritizes_primary(self):
        """Test that primary chunks are prioritized."""
        primary = [{"chunk_id": "1", "text": "test1", "vec": 0.7}]
        secondary = [{"chunk_id": "2", "text": "test2", "vec": 0.9}]
        result = _merge_and_deduplicate(primary, secondary, k=2)
        assert result[0]["chunk_id"] == "1"  # Primary comes first

