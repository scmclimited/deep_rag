"""
Legacy diagnostics.py - Maintained for backward compatibility.

This module now imports from the modularized diagnostics package.
"""
from retrieval.diagnostics import inspect_document, print_inspection_report

__all__ = ['inspect_document', 'print_inspection_report']
