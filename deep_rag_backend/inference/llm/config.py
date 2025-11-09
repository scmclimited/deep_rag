"""
LLM configuration and environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
DEFAULT_TEMP = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Gemini configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Future providers (commented out - uncomment when needed)
# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
# OLLAMA_URL = os.getenv("OLLAMA_URL", "")
# LLAVA_MODEL = os.getenv("LLAVA_MODEL", "llava-hf/llava-1.5-7b-hf")
# LLAVA_URL = os.getenv("LLAVA_URL", "")

