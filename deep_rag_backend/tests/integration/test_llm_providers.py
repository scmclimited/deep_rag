"""
Integration tests for LLM provider connectivity.
Tests connectivity with OpenAI, Google Gemini, and Ollama based on .env configuration.
"""
import pytest
import os
from dotenv import load_dotenv
from inference.llm import call_llm

load_dotenv()


class TestLLMProviderConnectivity:
    """Tests for LLM provider connectivity."""
    
    @pytest.mark.skipif(
        not os.getenv("LLM_PROVIDER"),
        reason="LLM_PROVIDER not set in environment"
    )
    def test_llm_provider_connectivity(self):
        """Test that the configured LLM provider is accessible."""
        llm_provider = os.getenv("LLM_PROVIDER", "").lower()
        
        # Simple test query
        test_message = [{"role": "user", "content": "Say 'test' if you can read this."}]
        
        try:
            response, token_info = call_llm(
                system="You are a helpful assistant.",
                messages=test_message,
                max_tokens=50,
                temperature=0.0
            )
            
            assert response is not None
            assert len(response) > 0
            assert isinstance(response, str)
            assert isinstance(token_info, dict)
            assert "total_tokens" in token_info
            
            print(f"✅ {llm_provider} connectivity test passed")
            print(f"Response: {response[:100]}...")
            
        except Exception as e:
            pytest.fail(f"LLM provider connectivity test failed: {e}")
