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
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        result = gemini_chat(
            system="Test system",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=100,
            temperature=0.2
        )
        assert result == "Test response"
    
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

