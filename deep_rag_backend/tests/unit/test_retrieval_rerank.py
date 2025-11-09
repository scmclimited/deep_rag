"""
Unit tests for reranking functionality.
"""
import pytest
from unittest.mock import patch, MagicMock
from retrieval.reranker.rerank import rerank_candidates


class TestRerankCandidates:
    """Tests for rerank_candidates function."""
    
    @patch('retrieval.reranker.rerank.get_reranker')
    def test_rerank_candidates_success(self, mock_get_reranker):
        """Test successful reranking."""
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.7, 0.5]
        mock_get_reranker.return_value = mock_reranker
        
        candidates = [
            {"chunk_id": "1", "text": "Candidate 1"},
            {"chunk_id": "2", "text": "Candidate 2"},
            {"chunk_id": "3", "text": "Candidate 3"}
        ]
        
        result = rerank_candidates("test query", candidates)
        
        assert len(result) == 3
        assert result[0]["chunk_id"] == "1"  # Highest score (0.9)
        assert result[0]["ce"] == 0.9
        assert result[1]["ce"] == 0.7
        assert result[2]["ce"] == 0.5
        mock_reranker.predict.assert_called_once()
    
    @patch('retrieval.reranker.rerank.get_reranker')
    def test_rerank_candidates_no_reranker(self, mock_get_reranker):
        """Test reranking when reranker is None."""
        mock_get_reranker.return_value = None
        
        candidates = [
            {"chunk_id": "1", "text": "Candidate 1"},
            {"chunk_id": "2", "text": "Candidate 2"}
        ]
        
        result = rerank_candidates("test query", candidates)
        
        # Should return candidates unchanged
        assert result == candidates
        assert "ce" not in result[0]  # No reranking scores added
    
    @patch('retrieval.reranker.rerank.get_reranker')
    def test_rerank_candidates_empty_list(self, mock_get_reranker):
        """Test reranking with empty candidates list."""
        mock_reranker = MagicMock()
        mock_get_reranker.return_value = mock_reranker
        
        result = rerank_candidates("test query", [])
        
        assert result == []
        mock_reranker.predict.assert_not_called()
    
    @patch('retrieval.reranker.rerank.get_reranker')
    def test_rerank_candidates_sorts_by_score(self, mock_get_reranker):
        """Test that candidates are sorted by reranking score."""
        mock_reranker = MagicMock()
        # Lower scores first in original order
        mock_reranker.predict.return_value = [0.3, 0.9, 0.5]
        mock_get_reranker.return_value = mock_reranker
        
        candidates = [
            {"chunk_id": "1", "text": "Low score"},
            {"chunk_id": "2", "text": "High score"},
            {"chunk_id": "3", "text": "Medium score"}
        ]
        
        result = rerank_candidates("test query", candidates)
        
        # Should be sorted by score (descending)
        assert result[0]["chunk_id"] == "2"  # Highest score (0.9)
        assert result[1]["chunk_id"] == "3"  # Medium score (0.5)
        assert result[2]["chunk_id"] == "1"  # Lowest score (0.3)
    
    @patch('retrieval.reranker.rerank.get_reranker')
    def test_rerank_candidates_handles_failure(self, mock_get_reranker):
        """Test that reranking failure returns original candidates."""
        mock_reranker = MagicMock()
        mock_reranker.predict.side_effect = Exception("Reranking failed")
        mock_get_reranker.return_value = mock_reranker
        
        candidates = [
            {"chunk_id": "1", "text": "Candidate 1"},
            {"chunk_id": "2", "text": "Candidate 2"}
        ]
        
        result = rerank_candidates("test query", candidates)
        
        # Should return original candidates on failure
        assert result == candidates
        assert "ce" not in result[0]  # No scores added on failure

