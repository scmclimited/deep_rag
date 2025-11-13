"""
Unit tests for Gemini LLM provider.
"""
import pytest
from unittest.mock import patch, MagicMock
from inference.llm.providers.gemini import gemini_chat
from inference.llm.config import GEMINI_API_KEY


class TestGeminiChat:
    """Tests for Gemini chat implementation."""
    
    @patch('inference.llm.providers.gemini.genai.Client')
    def test_gemini_chat_success(self, mock_client_class):
        """Test successful Gemini chat call."""
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 5
        mock_usage.total_token_count = 15
        mock_response.usage_metadata = mock_usage
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        result, token_info = gemini_chat(
            system="Test system",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=100,
            temperature=0.2
        )
        assert result == "Test response"
        assert token_info == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    
    def test_gemini_chat_no_api_key(self):
        """Test Gemini chat fails without API key."""
        with patch('inference.llm.providers.gemini.GEMINI_API_KEY', ''):
            with pytest.raises(EnvironmentError):
                gemini_chat(
                    system="Test system",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=100,
                    temperature=0.2
                )

