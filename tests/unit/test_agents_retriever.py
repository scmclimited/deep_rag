"""
Unit tests for retriever agent.
"""
import pytest
from unittest.mock import patch, MagicMock
from inference.agents.retriever import retriever_agent
from inference.agents.state import State


class TestRetrieverAgent:
    """Tests for retriever agent."""
    
    @patch('inference.agents.retriever.retrieve_hybrid')
    def test_retriever_basic(self, mock_retrieve):
        """Test basic retrieval functionality."""
        mock_retrieve.return_value = [
            {"chunk_id": "1", "text": "Evidence 1", "ce": 0.8, "lex": 0.5, "vec": 0.6, "p0": 1, "p1": 1, "doc_id": "doc1"}
        ]
        
        state: State = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = retriever_agent(state)
        
        assert len(result["evidence"]) == 1
        assert result["evidence"][0]["chunk_id"] == "1"
        mock_retrieve.assert_called_once()
    
    @patch('inference.agents.retriever.retrieve_hybrid')
    def test_retriever_with_doc_id(self, mock_retrieve):
        """Test retrieval with doc_id filter."""
        mock_retrieve.return_value = [
            {"chunk_id": "1", "text": "Evidence", "ce": 0.8, "lex": 0.5, "vec": 0.6, "p0": 1, "p1": 1, "doc_id": "doc1"}
        ]
        
        state: State = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_id": "doc1",
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = retriever_agent(state)
        
        # Verify doc_id is passed to retrieve_hybrid
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs.get("doc_id") == "doc1"
    
    @patch('inference.agents.retriever.retrieve_hybrid')
    def test_retriever_tracks_doc_ids(self, mock_retrieve):
        """Test that retriever tracks doc_ids from retrieved chunks."""
        mock_retrieve.return_value = [
            {"chunk_id": "1", "text": "Evidence 1", "ce": 0.8, "lex": 0.5, "vec": 0.6, "p0": 1, "p1": 1, "doc_id": "doc1"},
            {"chunk_id": "2", "text": "Evidence 2", "ce": 0.7, "lex": 0.4, "vec": 0.5, "p0": 2, "p1": 2, "doc_id": "doc2"}
        ]
        
        state: State = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = retriever_agent(state)
        
        # Should track doc_ids
        assert "doc1" in result["doc_ids"]
        assert "doc2" in result["doc_ids"]
    
    @patch('inference.agents.retriever.retrieve_hybrid')
    def test_retriever_with_cross_doc(self, mock_retrieve):
        """Test retrieval with cross_doc flag."""
        mock_retrieve.return_value = []
        
        state: State = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": True
        }
        
        result = retriever_agent(state)
        
        # Verify cross_doc is passed to retrieve_hybrid
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs.get("cross_doc") is True

