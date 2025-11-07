"""
LLM module - Unified interface for chat completion across providers.

Main entry point: call_llm()
"""
from inference.llm.wrapper import call_llm

__all__ = ['call_llm']

