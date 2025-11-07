"""
Google Gemini LLM provider implementation.
"""
import logging
from typing import List, Dict
from google import genai
from google.genai import types
from inference.llm.config import GEMINI_MODEL, GEMINI_API_KEY

logger = logging.getLogger(__name__)


def gemini_chat(
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
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY not set in environment. "
            "Get your key from https://makersuite.google.com/app/apikey"
        )
    
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
                        # Access usage_metadata as an object, not a dictionary
                        usage_metadata = getattr(response, 'usage_metadata', None)
                        if usage_metadata:
                            token_count = getattr(usage_metadata, 'total_token_count', 'unknown')
                        else:
                            token_count = 'unknown'
                        raise RuntimeError(
                            f"Gemini model {model_path} response was truncated due to MAX_TOKENS limit. "
                            f"Consider increasing max_tokens (currently {max_tokens}). "
                            f"Response had {token_count} tokens."
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
        error_msg += "\nNote: Using the 'google-genai' SDK. See https://github.com/googleapis/python-genai for documentation."
        raise RuntimeError(error_msg) from e

