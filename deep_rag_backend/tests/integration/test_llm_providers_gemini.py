"""
Integration tests for Gemini LLM provider connectivity.
"""
import pytest
import os
from dotenv import load_dotenv
from inference.llm import call_llm

load_dotenv()


class TestGeminiConnectivity:
    """Tests for Gemini provider connectivity."""
    
    @pytest.mark.skipif(
        os.getenv("LLM_PROVIDER", "").lower() != "gemini",
        reason="Gemini not configured"
    )
    def test_gemini_connectivity(self):
        """Test Gemini-specific connectivity."""
        test_message = [{"role": "user", "content": "Hello"}]
        response = call_llm(
            system="You are a helpful assistant.",
            messages=test_message,
            max_tokens=20
        )
        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)

