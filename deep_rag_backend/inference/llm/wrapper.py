"""
LLM wrapper - Unified interface for chat completion across providers.
"""
import time
import logging
from typing import List, Dict, Optional
from inference.llm.config import LLM_PROVIDER, DEFAULT_TEMP
from inference.llm.providers import gemini_chat

logger = logging.getLogger(__name__)

# Future providers (commented out - uncomment when needed)
# from inference.llm.providers import openai_chat, ollama_chat


def call_llm(
    system: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: Optional[float] = None,
    retries: int = 8,
    retry_backoff_sec: float = 2.0,
) -> tuple[str, Dict[str, int]]:
    """
    Unified interface for chat completion across providers.

    Args:
        system: system prompt string
        messages: list like [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
        max_tokens: max new tokens to generate
        temperature: sampling temperature; defaults from .env if None
        retries: retry attempts on transient errors
        retry_backoff_sec: exponential backoff base seconds

    Returns:
        assistant string response (stripped)
    """
    temperature = DEFAULT_TEMP if temperature is None else temperature
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            if LLM_PROVIDER == "gemini":
                text, token_info = gemini_chat(system, messages, max_tokens, temperature)
                return text, token_info
            # Future providers (commented out - uncomment when needed)
            # elif LLM_PROVIDER == "openai":
            #     return openai_chat(system, messages, max_tokens, temperature)
            # elif LLM_PROVIDER == "ollama":
            #     return ollama_chat(system, messages, max_tokens, temperature)
            else:
                raise ValueError(
                    f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. "
                    f"Currently only 'gemini' is supported. Set LLM_PROVIDER=gemini in .env"
                )
        except Exception as e:
            last_err = e
            if attempt == retries:
                break
            time.sleep(retry_backoff_sec * (2 ** (attempt - 1)))

    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")

