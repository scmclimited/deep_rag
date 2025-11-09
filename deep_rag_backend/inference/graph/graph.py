"""
Legacy graph.py - Maintained for backward compatibility.

This module now imports from the modularized graph package.
"""
from inference.graph.builder import build_app
from inference.graph.state import GraphState

__all__ = ['build_app', 'GraphState']
