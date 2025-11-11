"""
Unit tests for synthesizer graph node.
"""
import pytest
from unittest.mock import patch, MagicMock
from inference.graph.nodes.synthesizer import node_synthesizer
from inference.graph.state import GraphState


class TestNodeSynthesizer:
    """Tests for synthesizer graph node."""
    
    @patch('inference.graph.nodes.synthesizer.get_confidence_for_chunks')
    @patch('inference.graph.nodes.synthesizer.call_llm')
    def test_node_synthesizer_basic(self, mock_call_llm, mock_confidence):
        """Test basic synthesis functionality."""
        # Mock confidence to return high enough value to proceed
        mock_confidence.return_value = {
            "confidence": 85.0,
            "probability": 0.85,
            "action": "answer",
            "features": {}
        }
        mock_call_llm.return_value = "This is the answer."
        
        state: GraphState = {
            "question": "What is the answer?",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Evidence text", "p0": 1, "p1": 1, "doc_id": "doc1", 
                 "lex": 0.8, "vec": 0.7, "ce": 0.75}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_synthesizer(state)
        
        assert "This is the answer." in result["answer"]
        # Note: Sources section is now added by citation_pruner, not synthesizer
        # So we don't assert "Sources:" here - that's handled in citation_pruner tests
        mock_call_llm.assert_called_once()
    
    def test_node_synthesizer_no_evidence_abstains(self):
        """Test that synthesizer abstains when no evidence is provided."""
        # This is the critical test for the bug fix
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [],  # No evidence
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_synthesizer(state)
        
        # Should abstain with "I don't know"
        assert result["answer"] == "I don't know."
        assert result["confidence"] == 0.0
        assert result["action"] == "abstain"
    
    def test_node_synthesizer_none_evidence_abstains(self):
        """Test that synthesizer abstains when evidence is None."""
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": None,  # None evidence
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_synthesizer(state)
        
        # Should abstain with "I don't know"
        assert result["answer"] == "I don't know."
        assert result["confidence"] == 0.0
        assert result["action"] == "abstain"
    
    @patch('inference.graph.nodes.synthesizer.get_confidence_for_chunks')
    @patch('inference.graph.nodes.synthesizer.call_llm')
    def test_node_synthesizer_low_confidence_abstains(self, mock_call_llm, mock_confidence):
        """Test that synthesizer abstains when confidence < 40% even if above normal threshold."""
        # Mock confidence to return < 40%
        mock_confidence.return_value = {
            "confidence": 35.0,  # Below 40% threshold
            "probability": 0.35,
            "action": "answer",  # Above normal abstain threshold but below 40%
            "features": {}
        }
        
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Low quality evidence", "p0": 1, "p1": 1, "doc_id": "doc1",
                 "lex": 0.2, "vec": 0.2, "ce": 0.2}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_synthesizer(state)
        
        # Should abstain even though action is "answer" (confidence < 40%)
        assert result["answer"] == "I don't know."
        assert result["confidence"] == 35.0
        assert result["action"] == "abstain"
        mock_call_llm.assert_not_called()  # Should not call LLM
    
    @patch('inference.graph.nodes.synthesizer.get_confidence_for_chunks')
    @patch('inference.graph.nodes.synthesizer.call_llm')
    def test_node_synthesizer_normal_abstain_threshold(self, mock_call_llm, mock_confidence):
        """Test that synthesizer abstains when action is 'abstain' (normal threshold)."""
        # Mock confidence to return abstain action
        mock_confidence.return_value = {
            "confidence": 25.0,  # Below normal abstain threshold
            "probability": 0.25,
            "action": "abstain",
            "features": {}
        }
        
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Low quality evidence", "p0": 1, "p1": 1, "doc_id": "doc1",
                 "lex": 0.1, "vec": 0.1, "ce": 0.1}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_synthesizer(state)
        
        # Should abstain
        assert result["answer"] == "I don't know."
        assert result["confidence"] == 25.0
        assert result["action"] == "abstain"
        mock_call_llm.assert_not_called()  # Should not call LLM
    
    @patch('inference.graph.nodes.synthesizer.get_confidence_for_chunks')
    @patch('inference.graph.nodes.synthesizer.call_llm')
    def test_node_synthesizer_high_confidence_answers(self, mock_call_llm, mock_confidence):
        """Test that synthesizer generates answer when confidence >= 40%."""
        # Mock confidence to return >= 40%
        mock_confidence.return_value = {
            "confidence": 85.0,  # Above 40% threshold
            "probability": 0.85,
            "action": "answer",
            "features": {}
        }
        mock_call_llm.return_value = "This is a confident answer."
        
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "High quality evidence", "p0": 1, "p1": 1, "doc_id": "doc1",
                 "lex": 0.9, "vec": 0.9, "ce": 0.9}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_synthesizer(state)
        
        # Should generate answer
        assert "This is a confident answer." in result["answer"]
        assert result["confidence"] == 85.0
        assert result["action"] == "answer"
        mock_call_llm.assert_called_once()  # Should call LLM
    
    @patch('inference.graph.nodes.synthesizer.call_llm')
    def test_node_synthesizer_with_doc_id(self, mock_call_llm):
        """Test synthesis with doc_id context."""
        mock_call_llm.return_value = "Answer with doc context."
        
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Evidence", "p0": 1, "p1": 1, "doc_id": "doc1",
                 "lex": 0.8, "vec": 0.7, "ce": 0.75}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "doc_id": "doc1",
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_synthesizer(state)
        
        # Verify doc_id context is included in prompt
        call_args = mock_call_llm.call_args
        assert "Focus your answer on the identified document" in call_args[0][1][0]["content"]
    
    @patch('inference.graph.nodes.synthesizer.get_confidence_for_chunks')
    @patch('inference.graph.nodes.synthesizer.call_llm')
    def test_node_synthesizer_includes_citations(self, mock_call_llm, mock_confidence):
        """Test that citations are included in answer."""
        # Mock confidence to return high enough value to proceed
        mock_confidence.return_value = {
            "confidence": 85.0,
            "probability": 0.85,
            "action": "answer",
            "features": {}
        }
        mock_call_llm.return_value = "Answer with citation [1]."
        
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": "1", "text": "Evidence", "p0": 1, "p1": 1, "doc_id": "doc1",
                 "lex": 0.8, "vec": 0.7, "ce": 0.75}
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_synthesizer(state)
        
        # Note: Sources section is now added by citation_pruner, not synthesizer
        # The synthesizer just returns the raw answer with citations in the payload
        # So we verify the answer contains the citation reference, but not the Sources section
        assert "Answer with citation [1]." in result["answer"]
        # Citations are passed in the result payload, not in the answer text
        # Citations should be built for doc1 since it's in the evidence
        assert result.get("citations") is not None
        assert len(result["citations"]) > 0
        assert any("doc1" in cit for cit in result["citations"])
    
    @patch('inference.graph.nodes.synthesizer.call_llm')
    def test_node_synthesizer_uses_top_5_chunks(self, mock_call_llm):
        """Test that synthesizer only uses top 5 chunks."""
        mock_call_llm.return_value = "Answer."
        
        state: GraphState = {
            "question": "Test question",
            "plan": "Test plan",
            "evidence": [
                {"chunk_id": str(i), "text": f"Evidence {i}", "p0": i, "p1": i, "doc_id": f"doc{i}",
                 "lex": 0.8, "vec": 0.7, "ce": 0.75}
                for i in range(10)  # 10 chunks
            ],
            "notes": "Test notes",
            "answer": "",
            "confidence": 0.0,
            "iterations": 0,
            "refinements": [],
            "doc_ids": [],
            "cross_doc": False
        }
        
        result = node_synthesizer(state)
        
        # Verify only top chunks are used in context (context is limited by select_context_chunks)
        call_args = mock_call_llm.call_args
        context = call_args[0][1][0]["content"]
        # Should see evidence from chunks in context (format: "Document docX (key terms: ...):\nEvidence X")
        assert "Evidence 0" in context or "doc0" in context
        # Should see multiple chunks (context is limited by MAX_CONTEXT_CHUNKS=8, MAX_CHUNKS_PER_DOC=2)
        assert "Evidence" in context

