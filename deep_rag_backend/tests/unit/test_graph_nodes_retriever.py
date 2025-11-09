"""
Unit tests for retriever graph node.
"""
import pytest
from unittest.mock import patch, MagicMock
from inference.graph.nodes.retriever import node_retriever
from inference.graph.state import GraphState


class TestNodeRetriever:
    """Tests for retriever graph node."""
    
    @patch('inference.graph.nodes.retriever.retrieve_hybrid')
    def test_node_retriever_basic(self, mock_retrieve):
        """Test basic retrieval functionality with doc_id specified."""
        mock_retrieve.return_value = [
            {"chunk_id": "1", "text": "Evidence 1", "ce": 0.8, "lex": 0.5, "vec": 0.6, "p0": 1, "p1": 1, "doc_id": "doc1"}
        ]
        
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "doc_id": "doc1",  # Specify doc_id for retrieval
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_retriever(state)
        
        assert len(result["evidence"]) == 1
        assert result["evidence"][0]["chunk_id"] == "1"
        mock_retrieve.assert_called_once()
    
    @patch('inference.graph.nodes.retriever.retrieve_hybrid')
    def test_node_retriever_with_doc_id(self, mock_retrieve):
        """Test retrieval with doc_id filter."""
        mock_retrieve.return_value = [
            {"chunk_id": "1", "text": "Evidence", "ce": 0.8, "lex": 0.5, "vec": 0.6, "p0": 1, "p1": 1, "doc_id": "doc1"}
        ]
        
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "doc_id": "doc1",
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_retriever(state)
        
        # Verify doc_id is passed to retrieve_hybrid
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs.get("doc_id") == "doc1"
    
    @patch('inference.graph.nodes.retriever.retrieve_hybrid')
    def test_node_retriever_with_selected_doc_ids(self, mock_retrieve):
        """Test retrieval with selected_doc_ids (multi-document selection)."""
        mock_retrieve.return_value = [
            {"chunk_id": "1", "text": "Evidence 1", "ce": 0.8, "lex": 0.5, "vec": 0.6, "p0": 1, "p1": 1, "doc_id": "doc1"},
            {"chunk_id": "2", "text": "Evidence 2", "ce": 0.7, "lex": 0.4, "vec": 0.5, "p0": 2, "p1": 2, "doc_id": "doc2"},
            {"chunk_id": "3", "text": "Evidence 3", "ce": 0.6, "lex": 0.3, "vec": 0.4, "p0": 3, "p1": 3, "doc_id": "doc3"}
        ]
        
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "selected_doc_ids": ["doc1", "doc2"],
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_retriever(state)
        
        # Should only include chunks from selected doc_ids
        result_doc_ids = set(h.get('doc_id') for h in result["evidence"] if h.get('doc_id'))
        assert "doc1" in result_doc_ids
        assert "doc2" in result_doc_ids
        assert "doc3" not in result_doc_ids  # Should be filtered out
    
    @patch('inference.graph.nodes.retriever.retrieve_hybrid')
    def test_node_retriever_empty_selected_doc_ids(self, mock_retrieve):
        """Test that empty selected_doc_ids with cross_doc=False returns empty results."""
        # This is the critical test for the bug fix
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "selected_doc_ids": [],  # Empty - user deselected all documents
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_retriever(state)
        
        # Should return empty results without calling retrieve_hybrid
        assert len(result["evidence"]) == 0
        assert result["doc_ids"] == []
        mock_retrieve.assert_not_called()  # Should not call retrieve_hybrid
    
    @patch('inference.graph.nodes.retriever.retrieve_hybrid')
    def test_node_retriever_no_documents_specified(self, mock_retrieve):
        """Test that retriever returns empty when no documents specified and cross_doc=False."""
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_retriever(state)
        
        # Should return empty results without calling retrieve_hybrid
        assert len(result["evidence"]) == 0
        assert result["doc_ids"] == []
        mock_retrieve.assert_not_called()  # Should not call retrieve_hybrid
    
    @patch('inference.graph.nodes.retriever.retrieve_hybrid')
    def test_node_retriever_empty_selected_doc_ids_with_cross_doc(self, mock_retrieve):
        """Test that empty selected_doc_ids with cross_doc=True still searches all documents."""
        mock_retrieve.return_value = [
            {"chunk_id": "1", "text": "Evidence 1", "ce": 0.8, "lex": 0.5, "vec": 0.6, "p0": 1, "p1": 1, "doc_id": "doc1"}
        ]
        
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "selected_doc_ids": [],  # Empty but cross_doc=True
            "doc_ids": [],
            "cross_doc": True
        }
        
        result = node_retriever(state)
        
        # Should still retrieve (cross_doc=True ignores selected_doc_ids)
        assert len(result["evidence"]) == 1
        mock_retrieve.assert_called_once()
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs.get("cross_doc") is True
    
    @patch('inference.graph.nodes.retriever.retrieve_hybrid')
    def test_node_retriever_tracks_doc_ids(self, mock_retrieve):
        """Test that retriever tracks doc_ids from retrieved chunks."""
        mock_retrieve.return_value = [
            {"chunk_id": "1", "text": "Evidence 1", "ce": 0.8, "lex": 0.5, "vec": 0.6, "p0": 1, "p1": 1, "doc_id": "doc1"},
            {"chunk_id": "2", "text": "Evidence 2", "ce": 0.7, "lex": 0.4, "vec": 0.5, "p0": 2, "p1": 2, "doc_id": "doc2"}
        ]
        
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "selected_doc_ids": ["doc1", "doc2"],  # Specify selected_doc_ids for retrieval
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_retriever(state)
        
        # Should track doc_ids
        assert "doc1" in result["doc_ids"]
        assert "doc2" in result["doc_ids"]
    
    @patch('inference.graph.nodes.retriever.retrieve_hybrid')
    def test_node_retriever_filters_by_selected_doc_ids(self, mock_retrieve):
        """Test that retriever filters results to only selected doc_ids."""
        # Mock returns chunks from multiple docs
        mock_retrieve.return_value = [
            {"chunk_id": "1", "text": "Evidence 1", "ce": 0.8, "lex": 0.5, "vec": 0.6, "p0": 1, "p1": 1, "doc_id": "doc1"},
            {"chunk_id": "2", "text": "Evidence 2", "ce": 0.7, "lex": 0.4, "vec": 0.5, "p0": 2, "p1": 2, "doc_id": "doc2"},
            {"chunk_id": "3", "text": "Evidence 3", "ce": 0.6, "lex": 0.3, "vec": 0.4, "p0": 3, "p1": 3, "doc_id": "doc3"}
        ]
        
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "selected_doc_ids": ["doc1"],  # Only doc1 selected
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_retriever(state)
        
        # Should only include chunks from doc1
        assert len(result["evidence"]) == 1
        assert result["evidence"][0]["doc_id"] == "doc1"
        assert result["doc_ids"] == ["doc1"]

