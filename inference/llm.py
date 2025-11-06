# inference/llm.py
from __future__ import annotations
from typing import List, Dict, Optional
import os
import time
import logging
from dotenv import load_dotenv
# Future providers (commented out - currently using Gemini only):
# import torch          # For LLaVA model loading
# from PIL import Image # For LLaVA image processing
# import requests       # For Ollama HTTP API

load_dotenv()

logger = logging.getLogger(__name__)

# -------------------------
# LLM Provider Configuration
# Currently using Gemini only. Other providers commented out for future use.
# -------------------------
gemini_model = "gemini-2.5-flash-lite"  # or "gemini-2.5-pro" for latest, "gemini-1.5-flash" also supported

# Active provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # Currently defaulting to Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", gemini_model)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_TEMP = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Gemini client (currently active)
_GEMINI_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    _GEMINI_AVAILABLE = True
except Exception:
    pass

# Future providers (commented out - uncomment when needed)
# openai_model = "gpt-4o-mini"
# ollama_model = "llama3:8b"
# llava_model = "llava-hf/llava-1.5-7b-hf"
# OPENAI_MODEL = os.getenv("OPENAI_MODEL", openai_model)
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", ollama_model)
# LLAVA_MODEL = os.getenv("LLAVA_MODEL", llava_model)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# OLLAMA_URL = os.getenv("OLLAMA_URL", "")
# LLAVA_URL = os.getenv("LLAVA_URL", "")  # LLaVA service URL (if using service)


# Future providers (commented out - uncomment when needed)
# _OPENAI_AVAILABLE = False
# try:
#     from openai import OpenAI
#     _OPENAI_AVAILABLE = True
# except Exception:
#     pass
# 
# # Lazy loading for LLaVA model (only load when actually needed)
# _model = None
# _processor = None
# 
# def _load_llava_model():
#     """Lazy load LLaVA model and processor only when needed."""
#     global _model, _processor
#     if _model is None or _processor is None:
#         try:
#             import torch
#             from transformers import AutoProcessor, LlavaForConditionalGeneration
#             # Load the model in half-precision (recommended for VRAM efficiency)
#             # The model weights will download automatically the first time this code runs.
#             print("Loading LLaVA model and processor...")
#             _processor = AutoProcessor.from_pretrained(LLAVA_MODEL)
#             _model = LlavaForConditionalGeneration.from_pretrained(
#                 LLAVA_MODEL,
#                 torch_dtype=torch.float16,
#                 device_map="auto"
#             )
#             print("LLaVA model loaded successfully.")
#         except Exception as e:
#             error_msg = str(e)
#             if "ModelWrapper" in error_msg or "tokenizer" in error_msg.lower():
#                 raise RuntimeError(
#                     f"LLaVA model/tokenizer loading failed: {e}\n"
#                     f"This is often caused by corrupted model files or incompatibility.\n"
#                     f"RECOMMENDATION: Switch to 'ollama' or 'openai' provider for text-only chat.\n"
#                     f"LLaVA is designed for vision-language tasks and may not work well for text-only RAG.\n"
#                     f"To switch: Set LLM_PROVIDER=ollama in your .env file"
#                 ) from e
#             raise
#     return _model, _processor


# -------------------------
# Public entrypoint
# -------------------------
def call_llm(
    system: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: Optional[float] = None,
    retries: int = 8,
    retry_backoff_sec: float = 2.0,
) -> str:
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
                return _gemini_chat(system, messages, max_tokens, temperature)
            # Future providers (commented out - uncomment when needed)
            # elif LLM_PROVIDER == "openai":
            #     return _openai_chat(system, messages, max_tokens, temperature)
            # elif LLM_PROVIDER == "ollama":
            #     return _ollama_chat(system, messages, max_tokens, temperature)
            # elif LLM_PROVIDER == "llava":
            #     return _llava_chat(system, messages, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Currently only 'gemini' is supported. Set LLM_PROVIDER=gemini in .env")
        except Exception as e:
            last_err = e
            if attempt == retries:
                break
            time.sleep(retry_backoff_sec * (2 ** (attempt - 1)))

    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")

# -------------------------
# Provider implementations
# -------------------------

# Future LLM providers (commented out - uncomment when needed)

# def _llava_chat(
#     system: str,
#     messages: List[Dict[str, str]],
#     max_tokens: int,
#     temperature: float
# ) -> str:
#     """
#     LLaVA chat implementation using locally loaded model.
#     Loads the model lazily on first use.
#     
#     Note: LLaVA is a vision-language model. For text-only chat,
#     this uses the model's text generation capabilities.
#     
#     WARNING: LLaVA is designed for vision-language tasks and may not work well
#     for text-only chat. Consider using 'ollama' or 'openai' providers instead.
#     """
#     try:
#         # Load model if not already loaded
#         model, processor = _load_llava_model()
#         
#         # Compose prompt from system and messages
#         # LLaVA expects a conversation format
#         prompt_parts = []
#         if system:
#             prompt_parts.append(f"System: {system}")
#         
#         for msg in messages:
#             role = msg.get("role", "user")
#             content = msg.get("content", "")
#             if role == "user":
#                 prompt_parts.append(f"USER: {content}")
#             elif role == "assistant":
#                 prompt_parts.append(f"ASSISTANT: {content}")
#         
#         prompt_parts.append("ASSISTANT:")
#         full_prompt = "\n".join(prompt_parts)
#         
#         # LLaVA processor expects images, but we can pass None/empty for text-only
#         tokenizer = processor.tokenizer
#         
#         # Tokenize the prompt
#         inputs = tokenizer(
#             full_prompt,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=2048  # Reasonable limit
#         ).to(model.device)
#         
#         # For LLaVA, we need to provide image inputs even if empty
#         # Create a dummy image (single pixel) as LLaVA requires image input
#         dummy_image = Image.new('RGB', (1, 1), color='white')
#         pixel_values = processor.image_processor(
#             dummy_image,
#             return_tensors="pt"
#         )["pixel_values"].to(model.device)
#         
#         # Generate response
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 pixel_values=pixel_values,
#                 max_new_tokens=max_tokens,
#                 temperature=temperature if temperature > 0 else 0.7,
#                 do_sample=temperature > 0,
#                 pad_token_id=tokenizer.eos_token_id
#             )
#         
#         # Decode the response (skip the input tokens)
#         input_length = inputs["input_ids"].shape[1]
#         generated_text = tokenizer.batch_decode(
#             outputs[:, input_length:],
#             skip_special_tokens=True
#         )[0]
#         
#         return generated_text.strip()
#     except Exception as e:
#         # If LLaVA fails, suggest using Ollama or OpenAI instead
#         raise RuntimeError(
#             f"LLaVA model failed: {e}\n"
#             f"LLaVA is a vision-language model and may not work well for text-only chat.\n"
#             f"Consider switching to 'ollama' or 'openai' provider by setting LLM_PROVIDER in .env"
#         ) from e
# 
# def _openai_chat(
#     system: str,
#     messages: List[Dict[str, str]],
#     max_tokens: int,
#     temperature: float
# ) -> str:
#     if not _OPENAI_AVAILABLE:
#         raise ImportError("openai package is not installed. Add `openai>=1.35.0` to requirements.")
#     if not OPENAI_API_KEY:
#         raise EnvironmentError("OPENAI_API_KEY not set in environment.")
# 
#     client = OpenAI(api_key=OPENAI_API_KEY)
# 
#     # Compose messages with system first
#     msg_payload = [{"role": "system", "content": system}] + messages
# 
#     # Use the Chat Completions API
#     resp = client.chat.completions.create(
#         model=OPENAI_MODEL,
#         messages=msg_payload,
#         max_tokens=max_tokens,
#         temperature=temperature,
#     )
#     return (resp.choices[0].message.content or "").strip()
# 
# # def _ollama_chat(
#     system: str,
#     messages: List[Dict[str, str]],
#     max_tokens: int,
#     temperature: float
# ) -> str:
#     """
#     Talks to Ollama's /api/chat endpoint.
#     You must have the model pulled locally:
#       ollama pull llama3:8b
#       ollama serve
#     """
#     import requests
#     url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
#     msg_payload = [{"role": "system", "content": system}] + messages
# 
#     data = {
#         "model": OLLAMA_MODEL,
#         "messages": msg_payload,
#         "options": {
#             "temperature": temperature,
#             "num_predict": max_tokens
#         },
#         "stream": False
#     }
#     r = requests.post(url, json=data, timeout=300)
#     r.raise_for_status()
#     js = r.json()
#     # Response shape: {"message": {"role":"assistant","content":"..."} , ...}
#     content = js.get("message", {}).get("content", "")
#     return (content or "").strip()

# Active provider implementation (currently using Gemini)
def _gemini_chat(
    system: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float
) -> str:
    """
    Gemini chat implementation using Google's new GenAI SDK (google-genai).
    Based on: https://github.com/googleapis/python-genai
    
    Gemini is multi-modal (text, images, audio, video) but this implementation
    currently handles text-only. Can be extended for multi-modal later.
    
    Requires GEMINI_API_KEY to be set in environment.
    """
    if not _GEMINI_AVAILABLE:
        raise ImportError("google-genai package is not installed. Add `google-genai>=1.0.0` to requirements.")
    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY not set in environment. Get your key from https://makersuite.google.com/app/apikey")
    
    # Use the configured model directly - no need to check available models
    # The model name is set in .env file (GEMINI_MODEL)
    model_name = GEMINI_MODEL
    logger.debug(f"Using Gemini model: {model_name}")
    
    # Build config for the new SDK
    # System instruction is set in config
    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    if system:
        config.system_instruction = system
    
    # Build contents from messages
    # The new SDK expects a list of Content objects or a simple string
    # For simplicity, we'll combine messages into a single string for now
    # The SDK can handle multi-turn conversations with Content objects later
    content_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            content_parts.append(content)
        elif role == "assistant":
            # For assistant messages, we'd use chat history, but for simplicity we'll include them
            content_parts.append(f"\nAssistant: {content}\n")
    
    # Combine into a single string for the user message
    user_content = "\n".join(content_parts).strip()
    
    # Use the configured model directly - format as "models/{model_name}"
    model_path = f"models/{model_name}" if not model_name.startswith("models/") else model_name
    
    try:
        # Use the new SDK's generate_content method
        with genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(api_version='v1alpha')) as client:
            response = client.models.generate_content(
                model=model_path,
                contents=user_content,
                config=config
            )
        
        # Try to get text directly from response.text property first
        # Note: response.text is a computed property that extracts from candidates[0].content.parts
        # If parts is None (e.g., MAX_TOKENS), text might be None even if hasattr returns True
        if hasattr(response, 'text'):
            try:
                text_value = response.text
                # Check if text_value is not None and not empty
                if text_value and str(text_value).strip():
                    return str(text_value).strip()
            except (AttributeError, TypeError, Exception) as e:
                logger.debug(f"Could not access response.text: {e}")
                pass
        
        # Fallback: Try to extract text from candidates structure
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            # Check if candidate has content with parts
            if (hasattr(candidate, 'content') and candidate.content is not None and 
                hasattr(candidate.content, 'parts') and candidate.content.parts is not None):
                text_parts = []
                for part in candidate.content.parts:
                    if part is not None and hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    return " ".join(text_parts).strip()
            
            # Handle case where parts is None (e.g., MAX_TOKENS finish_reason)
            # When parts is None, the response might still have generated text before hitting the limit
            if hasattr(candidate, 'content') and candidate.content is not None:
                finish_reason = getattr(candidate, 'finish_reason', None)
                
                # If parts is None, try alternative ways to get text
                if getattr(candidate.content, 'parts', None) is None:
                    logger.warning(f"Response content.parts is None. Finish reason: {finish_reason}")
                    
                    # Try to access text directly on content if available
                    if hasattr(candidate.content, 'text') and candidate.content.text:
                        return candidate.content.text.strip()
                    
                    # If finish_reason is MAX_TOKENS, the response was truncated
                    # In this case, we might need to return an error or partial response
                    if finish_reason and 'MAX_TOKENS' in str(finish_reason):
                        raise RuntimeError(
                            f"Gemini model {model_path} response was truncated due to MAX_TOKENS limit. "
                            f"Consider increasing max_tokens (currently {max_tokens}). "
                            f"Response had {getattr(response, 'usage_metadata', {}).get('total_token_count', 'unknown')} tokens."
                        )

        # Log response structure for debugging if we get here
        logger.error(f"Unexpected response structure. Response type: {type(response)}, "
                    f"Has text: {hasattr(response, 'text')}, "
                    f"Text value: {getattr(response, 'text', None) if hasattr(response, 'text') else 'N/A'}, "
                    f"Has candidates: {hasattr(response, 'candidates')}, "
                    f"Candidates: {getattr(response, 'candidates', None)}")
        raise RuntimeError(f"Gemini model {model_path} returned empty or unexpected response structure")
        
    except Exception as e:
        # Provide helpful error message
        error_msg = f"Gemini API call failed with model {model_path}: {e}\n"
        error_msg += f"Configured model: {GEMINI_MODEL}\n"
        error_msg += "\nNote: Using the new 'google-genai' SDK. See https://github.com/googleapis/python-genai for documentation."
        raise RuntimeError(error_msg) from e
