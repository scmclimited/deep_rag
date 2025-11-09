"""
Query command - Query existing documents.
"""
import typer
from typing import Optional
from inference.agents import run_deep_rag


def query(
    question: str = typer.Argument(..., help="The question to ask against ingested documents"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", "-d", help="Optional document ID (UUID) to filter retrieval to a specific document"),
    cross_doc: bool = typer.Option(False, "--cross-doc", help="Enable cross-document retrieval (two-stage when doc_id provided)")
):
    """
    Query existing documents in the vector database.
    Assumes documents have already been ingested.
    
    If --doc-id is provided, retrieval is filtered to that specific document.
    If --doc-id is not provided, retrieval searches across all documents.
    If --cross-doc is enabled, performs two-stage retrieval (doc_id first, then cross-doc semantic search).
    
    Matches: POST /ask endpoint
    """
    try:
        if doc_id:
            typer.echo(f"🔍 Querying with document filter: {doc_id}...")
        if cross_doc:
            typer.echo("🌐 Cross-document retrieval enabled")
        answer = run_deep_rag(question, doc_id=doc_id, cross_doc=cross_doc)
        typer.echo("\n" + "="*80)
        typer.echo("Answer:")
        typer.echo("="*80)
        typer.echo(answer)
        typer.echo("="*80)
    except Exception as e:
        typer.echo(f"Error querying: {e}", err=True)
        raise typer.Exit(1)

