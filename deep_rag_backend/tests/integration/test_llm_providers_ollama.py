"""
Integration tests for Ollama LLM provider connectivity.
"""
import pytest
import os
from dotenv import load_dotenv
from inference.llm import call_llm

load_dotenv()


class TestOllamaConnectivity:
    """Tests for Ollama provider connectivity."""
    
    @pytest.mark.skipif(
        os.getenv("LLM_PROVIDER", "").lower() != "ollama",
        reason="Ollama not configured"
    )
    def test_ollama_connectivity(self):
        """Test Ollama-specific connectivity."""
        test_message = [{"role": "user", "content": "Hello"}]
        response = call_llm(
            system="You are a helpful assistant.",
            messages=test_message,
            max_tokens=20
        )
        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)

