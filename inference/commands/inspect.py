"""
Inspect command - Inspect document chunks and pages.
"""
import typer
from typing import Optional
from retrieval.diagnostics import print_inspection_report


def inspect(
    doc_title: Optional[str] = typer.Option(None, "--title", "-t", help="Document title to search for (partial match)"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", "-d", help="Document ID (UUID)")
):
    """
    Inspect what chunks and pages are stored for a document.
    Useful for debugging ingestion and retrieval issues.
    
    Shows:
    - Total chunks and pages stored
    - Page distribution with chunk counts
    - Sample text from each page
    
    Matches: GET /diagnostics/document endpoint
    """
    try:
        print_inspection_report(doc_title=doc_title, doc_id=doc_id)
    except Exception as e:
        typer.echo(f"Error inspecting document: {e}", err=True)
        raise typer.Exit(1)

