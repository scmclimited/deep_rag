"""
Unit tests for compressor agent.
"""
import pytest
from unittest.mock import patch, MagicMock
from inference.agents.compressor import compressor
from inference.agents.state import State


class TestCompressor:
    """Tests for compressor agent."""
    
    @patch('inference.agents.compressor.call_llm')
    def test_compressor_basic(self, mock_call_llm):
        """Test basic compression functionality."""
        mock_call_llm.return_value = "- Key point 1\n- Key point 2"
        
        state: State = {
            "question": "Test question",
            "plan": "",
            "evidence": [
                {"chunk_id": "1", "text": "Evidence text 1", "p0": 1, "p1": 1},
                {"chunk_id": "2", "text": "Evidence text 2", "p0": 2, "p1": 2}
            ],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = compressor(state)
        
        assert result["notes"] == "- Key point 1\n- Key point 2"
        mock_call_llm.assert_called_once()
    
    @patch('inference.agents.compressor.call_llm')
    def test_compressor_empty_evidence(self, mock_call_llm):
        """Test compression with empty evidence."""
        mock_call_llm.return_value = "No evidence found"
        
        state: State = {
            "question": "Test question",
            "plan": "",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = compressor(state)
        
        assert result["notes"] == "No evidence found"
        mock_call_llm.assert_called_once()
    
    @patch('inference.agents.compressor.call_llm')
    def test_compressor_truncates_long_text(self, mock_call_llm):
        """Test that long evidence text is truncated."""
        mock_call_llm.return_value = "Compressed notes"
        
        long_text = "A" * 2000  # Longer than 1200 char limit
        state: State = {
            "question": "Test question",
            "plan": "",
            "evidence": [
                {"chunk_id": "1", "text": long_text, "p0": 1, "p1": 1}
            ],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = compressor(state)
        
        # Verify that text was truncated in the prompt
        call_args = mock_call_llm.call_args
        prompt_text = call_args[0][1][0]["content"]
        assert len(prompt_text) < len(long_text) + 500  # Should be truncated
    
    @patch('inference.agents.compressor.call_llm')
    def test_compressor_strips_whitespace(self, mock_call_llm):
        """Test that notes are stripped of whitespace."""
        mock_call_llm.return_value = "  \n  Notes with whitespace  \n  "
        
        state: State = {
            "question": "Test question",
            "plan": "",
            "evidence": [{"chunk_id": "1", "text": "Evidence", "p0": 1, "p1": 1}],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = compressor(state)
        
        assert result["notes"] == "Notes with whitespace"

