"""
Unit tests for critic agent.
"""
import pytest
from unittest.mock import patch, MagicMock
from inference.agents.critic import critic
from inference.agents.state import State
from inference.agents.constants import MAX_ITERS, THRESH


class TestCritic:
    """Tests for critic agent."""
    
    @patch('inference.agents.critic.retrieve_hybrid')
    @patch('inference.agents.critic.call_llm')
    def test_critic_high_confidence(self, mock_call_llm, mock_retrieve):
        """Test critic with high confidence (no refinement needed)."""
        # Need multiple strong chunks to get confidence >= 0.6
        # conf = min(0.9, 0.4 + 0.1*strong), so need strong >= 2
        state: State = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Evidence 1", "ce": 0.8, "lex": 0.5, "vec": 0.6, "p0": 1, "p1": 1},
                {"chunk_id": "2", "text": "Evidence 2", "ce": 0.7, "lex": 0.4, "vec": 0.5, "p0": 2, "p1": 2},
                {"chunk_id": "3", "text": "Evidence 3", "ce": 0.6, "lex": 0.3, "vec": 0.4, "p0": 3, "p1": 3}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = critic(state)
        
        # High confidence should not trigger refinement
        assert result["confidence"] >= 0.6
        mock_call_llm.assert_not_called()  # No refinement query
        mock_retrieve.assert_not_called()
    
    @patch('inference.agents.critic.retrieve_hybrid')
    @patch('inference.agents.critic.call_llm')
    def test_critic_low_confidence_triggers_refinement(self, mock_call_llm, mock_retrieve):
        """Test critic with low confidence triggers refinement."""
        # Mock returns strong chunks after refinement to stop recursion
        mock_call_llm.return_value = "What are the specific requirements?"
        # First call returns weak chunks, subsequent calls return strong chunks to stop recursion
        mock_retrieve.side_effect = [
            [{"chunk_id": "2", "text": "New evidence", "ce": 0.3, "lex": 0.2, "vec": 0.3, "p0": 2, "p1": 2}],
            [{"chunk_id": "3", "text": "Strong evidence", "ce": 0.8, "lex": 0.5, "vec": 0.6, "p0": 3, "p1": 3},
             {"chunk_id": "4", "text": "Strong evidence 2", "ce": 0.7, "lex": 0.4, "vec": 0.5, "p0": 4, "p1": 4}]
        ]
        
        state: State = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Weak evidence", "ce": 0.2, "lex": 0.0, "vec": 0.0, "p0": 1, "p1": 1}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = critic(state)
        
        # Should have triggered refinement (recursive calls until confidence >= 0.6 or max iterations)
        assert result["iterations"] >= 1
        assert len(result["evidence"]) >= 1
        assert mock_call_llm.call_count >= 1  # At least one refinement query
        assert mock_retrieve.call_count >= 1  # At least one additional retrieval
    
    @patch('inference.agents.critic.retrieve_hybrid')
    @patch('inference.agents.critic.call_llm')
    def test_critic_max_iterations(self, mock_call_llm, mock_retrieve):
        """Test critic stops at max iterations."""
        state: State = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Weak evidence", "ce": 0.2, "lex": 0.0, "vec": 0.0, "p0": 1, "p1": 1}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": MAX_ITERS,  # Already at max
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = critic(state)
        
        # Should not trigger refinement (already at max iterations)
        assert result["iterations"] == MAX_ITERS
        mock_call_llm.assert_not_called()
        mock_retrieve.assert_not_called()
    
    @patch('inference.agents.critic.retrieve_hybrid')
    @patch('inference.agents.critic.call_llm')
    def test_critic_with_doc_id(self, mock_call_llm, mock_retrieve):
        """Test critic with doc_id filter."""
        # Use UUID format for doc_id
        test_doc_id = "550e8400-e29b-41d4-a716-446655440000"
        mock_call_llm.return_value = "Refinement query"
        # Return strong chunks after first refinement to stop recursion
        mock_retrieve.side_effect = [
            [],
            [{"chunk_id": "2", "text": "Strong evidence", "ce": 0.8, "lex": 0.5, "vec": 0.6, "p0": 2, "p1": 2, "doc_id": test_doc_id},
             {"chunk_id": "3", "text": "Strong evidence 2", "ce": 0.7, "lex": 0.4, "vec": 0.5, "p0": 3, "p1": 3, "doc_id": test_doc_id}]
        ]
        
        state: State = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Weak evidence", "ce": 0.2, "lex": 0.0, "vec": 0.0, "p0": 1, "p1": 1}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_id": test_doc_id,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = critic(state)
        
        # Verify doc_id is passed to retrieve_hybrid (check first call)
        assert mock_retrieve.call_count >= 1
        first_call_kwargs = mock_retrieve.call_args_list[0][1]
        assert first_call_kwargs.get("doc_id") == test_doc_id
    
    @patch('inference.agents.critic.retrieve_hybrid')
    @patch('inference.agents.critic.call_llm')
    def test_critic_deduplicates_evidence(self, mock_call_llm, mock_retrieve):
        """Test that critic deduplicates evidence by chunk_id."""
        mock_call_llm.return_value = "Refinement query"
        mock_retrieve.return_value = [
            {"chunk_id": "1", "text": "Duplicate", "ce": 0.3, "lex": 0.2, "vec": 0.3, "p0": 1, "p1": 1}
        ]
        
        state: State = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Original", "ce": 0.2, "lex": 0.0, "vec": 0.0, "p0": 1, "p1": 1}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = critic(state)
        
        # Should deduplicate - only one chunk with chunk_id "1"
        chunk_ids = [chunk["chunk_id"] for chunk in result["evidence"]]
        assert chunk_ids.count("1") == 1

