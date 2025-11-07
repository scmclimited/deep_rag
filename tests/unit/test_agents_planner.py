"""
Unit tests for planner agent.
"""
import pytest
from unittest.mock import patch, MagicMock
from inference.agents.planner import planner
from inference.agents.state import State


class TestPlanner:
    """Tests for planner agent."""
    
    @patch('inference.agents.planner.call_llm')
    def test_planner_basic(self, mock_call_llm):
        """Test basic planning functionality."""
        mock_call_llm.return_value = "1. Find main topics\n2. Identify key points"
        
        state: State = {
            "question": "What is the document about?",
            "plan": "",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = planner(state)
        
        assert result["plan"] == "1. Find main topics\n2. Identify key points"
        mock_call_llm.assert_called_once()
    
    @patch('inference.agents.planner.call_llm')
    def test_planner_with_doc_id(self, mock_call_llm):
        """Test planning with doc_id context."""
        mock_call_llm.return_value = "1. Analyze document content"
        
        state: State = {
            "question": "What is in this document?",
            "plan": "",
            "evidence": [],
            "notes": "",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_id": "test-doc-id-123",
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = planner(state)
        
        assert result["plan"] == "1. Analyze document content"
        # Verify doc_id context is included in prompt
        call_args = mock_call_llm.call_args
        assert "specific document" in call_args[0][1][0]["content"]
    
    @patch('inference.agents.planner.call_llm')
    def test_planner_strips_whitespace(self, mock_call_llm):
        """Test that plan is stripped of whitespace."""
        mock_call_llm.return_value = "  \n  Plan with whitespace  \n  "
        
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
        
        result = planner(state)
        
        assert result["plan"] == "Plan with whitespace"
    
    @patch('inference.agents.planner.call_llm')
    def test_planner_llm_failure(self, mock_call_llm):
        """Test handling of LLM call failure."""
        mock_call_llm.side_effect = Exception("LLM API error")
        
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
        
        with pytest.raises(Exception, match="LLM API error"):
            planner(state)

