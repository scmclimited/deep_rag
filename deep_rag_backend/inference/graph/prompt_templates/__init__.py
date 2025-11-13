"""
Prompt templates for graph nodes.

This module provides access to all prompt templates used by the graph nodes.
Each template is stored in a separate file for easy maintenance and updates.
"""
from pathlib import Path

# Base directory for templates
TEMPLATES_DIR = Path(__file__).parent


def load_template(template_name: str) -> str:
    """
    Load a prompt template from a file.
    
    Args:
        template_name: Name of the template file (without .txt extension)
        
    Returns:
        Template string content
    """
    template_path = TEMPLATES_DIR / f"{template_name}.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Template '{template_name}' not found at {template_path}")
    
    return template_path.read_text(encoding='utf-8')


def format_template(template_name: str, **kwargs) -> str:
    """
    Load and format a prompt template with provided variables.
    
    Args:
        template_name: Name of the template file (without .txt extension)
        **kwargs: Variables to format into the template
        
    Returns:
        Formatted template string
    """
    template = load_template(template_name)
    return template.format(**kwargs)

