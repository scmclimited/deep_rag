"""
Unit tests for synthesizer agent.
"""
import pytest
from unittest.mock import patch, MagicMock
from inference.agents.synthesizer import synthesizer
from inference.agents.state import State


class TestSynthesizer:
    """Tests for synthesizer agent."""
    
    @patch('inference.agents.synthesizer.call_llm')
    def test_synthesizer_basic(self, mock_call_llm):
        """Test basic synthesis functionality."""
        mock_call_llm.return_value = "This is the answer."
        
        state: State = {
            "question": "What is the answer?",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Evidence text", "p0": 1, "p1": 1, "doc_id": "doc1"}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = synthesizer(state)
        
        assert "This is the answer." in result["answer"]
        assert "Sources:" in result["answer"]
        mock_call_llm.assert_called_once()
    
    @patch('inference.agents.synthesizer.call_llm')
    def test_synthesizer_with_doc_id(self, mock_call_llm):
        """Test synthesis with doc_id context."""
        mock_call_llm.return_value = "Answer with doc context."
        
        state: State = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Evidence", "p0": 1, "p1": 1, "doc_id": "doc1"}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_id": "doc1",
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = synthesizer(state)
        
        # Verify doc_id context is included in prompt
        call_args = mock_call_llm.call_args
        assert "specific document" in call_args[0][1][0]["content"]
    
    @patch('inference.agents.synthesizer.call_llm')
    def test_synthesizer_includes_citations(self, mock_call_llm):
        """Test that citations are included in answer."""
        mock_call_llm.return_value = "Answer with citation [1]."
        
        state: State = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Evidence", "p0": 1, "p1": 1, "doc_id": "doc1"}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = synthesizer(state)
        
        # Should include Sources section with citation
        assert "Sources:" in result["answer"]
        assert "doc:doc1" in result["answer"] or "p1" in result["answer"]
    
    @patch('inference.agents.synthesizer.call_llm')
    def test_synthesizer_tracks_doc_ids(self, mock_call_llm):
        """Test that synthesizer tracks doc_ids from evidence."""
        mock_call_llm.return_value = "Answer."
        
        state: State = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Evidence 1", "p0": 1, "p1": 1, "doc_id": "doc1"},
                {"chunk_id": "2", "text": "Evidence 2", "p0": 2, "p1": 2, "doc_id": "doc2"}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = synthesizer(state)
        
        # Should track doc_ids
        assert "doc1" in result["doc_ids"]
        assert "doc2" in result["doc_ids"]
    
    @patch('inference.agents.synthesizer.call_llm')
    def test_synthesizer_uses_top_5_chunks(self, mock_call_llm):
        """Test that synthesizer only uses top 5 chunks."""
        mock_call_llm.return_value = "Answer."
        
        state: State = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": str(i), "text": f"Evidence {i}", "p0": i, "p1": i, "doc_id": f"doc{i}"}
                for i in range(10)  # 10 chunks
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = synthesizer(state)
        
        # Verify only top 5 chunks are used in context
        call_args = mock_call_llm.call_args
        context = call_args[0][1][0]["content"]
        # Should only see chunks 1-5 in context
        assert "[1]" in context
        assert "[5]" in context
        assert "[6]" not in context  # Should not be included

