"""
Query graph command - Query existing documents using LangGraph.
"""
import typer
from typing import Optional
from inference.graph.graph_wrapper import ask_with_graph


def query_graph(
    question: str = typer.Argument(..., help="The question to ask against ingested documents (uses LangGraph)"),
    thread_id: str = typer.Option("default", "--thread-id", help="Thread ID for conversation state"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", "-d", help="Optional document ID (UUID) to filter retrieval to a specific document"),
    cross_doc: bool = typer.Option(False, "--cross-doc", help="Enable cross-document retrieval (two-stage when doc_id provided)")
):
    """
    Query existing documents using LangGraph pipeline with conditional routing.
    
    The graph allows agents to decide if they have sufficient evidence
    or need to iterate over query refinement and refine_retrieve options.
    
    If --doc-id is provided, retrieval is filtered to that specific document.
    If --doc-id is not provided, retrieval searches across all documents.
    If --cross-doc is enabled, performs two-stage retrieval (doc_id first, then cross-doc semantic search).
    
    Matches: POST /ask-graph endpoint
    """
    try:
        if doc_id:
            typer.echo(f"🔍 Querying with document filter: {doc_id}...")
        if cross_doc:
            typer.echo("🌐 Cross-document retrieval enabled")
        answer = ask_with_graph(question, thread_id=thread_id, doc_id=doc_id, cross_doc=cross_doc)
        typer.echo("\n" + "="*80)
        typer.echo("Answer:")
        typer.echo("="*80)
        typer.echo(answer)
        typer.echo("="*80)
    except Exception as e:
        typer.echo(f"Error querying with graph: {e}", err=True)
        raise typer.Exit(1)

