"""
Unit tests for LLM wrapper module.
"""
import pytest
from unittest.mock import patch, MagicMock
from inference.llm.wrapper import call_llm
from inference.llm.config import LLM_PROVIDER


class TestCallLLM:
    """Tests for call_llm wrapper function."""
    
    @patch('inference.llm.wrapper.gemini_chat')
    def test_call_llm_gemini(self, mock_gemini):
        """Test call_llm with Gemini provider."""
        mock_gemini.return_value = ("Test response", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        result, token_info = call_llm(
            system="Test system",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=100
        )
        assert result == "Test response"
        assert token_info["total_tokens"] == 15
        mock_gemini.assert_called_once()
    
    @patch('inference.llm.wrapper.gemini_chat')
    def test_call_llm_retry_on_error(self, mock_gemini):
        """Test call_llm retries on transient errors."""
        mock_gemini.side_effect = [Exception("Transient error"), ("Success", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})]
        result, token_info = call_llm(
            system="Test system",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=100,
            retries=2
        )
        assert result == "Success"
        assert mock_gemini.call_count == 2

