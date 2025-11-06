import typer
from pathlib import Path
from typing import Optional
from ingestion.ingest import ingest as ingest_pdf
from ingestion.ingest_text import ingest_text_file
from ingestion.ingest_image import ingest_image
from inference.agent_loop import run_deep_rag
from inference.graph.graph_wrapper import ask_with_graph

app = typer.Typer(help="Deep RAG CLI - matches FastAPI service routes")

@app.command()
def ingest(
    file: str,
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Custom title for the document. If not provided, will be extracted from file metadata or filename.")
):
    """
    Ingest a document into the vector database without querying.
    Supports PDF, TXT, PNG, JPEG.
    
    This command only handles embedding and upsert - no agentic reasoning.
    Use 'infer' command for ingestion + query with agentic reasoning.
    
    Matches: POST /ingest endpoint
    """
    file_path = Path(file)
    if not file_path.exists():
        typer.echo(f"Error: File not found: {file}", err=True)
        raise typer.Exit(1)
    
    file_ext = file_path.suffix.lower()
    
    try:
        if file_ext == '.pdf':
            ingest_pdf(str(file_path), title=title)
            typer.echo(f"✅ Ingested PDF: {file_path.name}")
        elif file_ext == '.txt':
            ingest_text_file(str(file_path), title=title or file_path.stem)
            typer.echo(f"✅ Ingested text file: {file_path.name}")
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            ingest_image(str(file_path), title=title or file_path.stem)
            typer.echo(f"✅ Ingested image: {file_path.name}")
        else:
            typer.echo(f"Error: Unsupported file type: {file_ext}. Supported: PDF, TXT, PNG, JPEG", err=True)
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error ingesting file: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask against ingested documents")
):
    """
    Query existing documents in the vector database.
    Assumes documents have already been ingested.
    
    Matches: POST /ask endpoint
    """
    try:
        answer = run_deep_rag(question)
        typer.echo("\n" + "="*80)
        typer.echo("Answer:")
        typer.echo("="*80)
        typer.echo(answer)
        typer.echo("="*80)
    except Exception as e:
        typer.echo(f"Error querying: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def infer(
    question: str = typer.Argument(..., help="The question to ask"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Optional file to ingest before querying (PDF, TXT, PNG, JPEG)"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Custom title for the document (only used if --file is provided)")
):
    """
    Combined ingestion and query endpoint.
    
    If --file is provided:
    - Ingest the file (PDF, TXT, PNG, JPEG)
    - Then query using the question
    
    If no --file:
    - Just query existing documents (same as 'query' command)
    
    Matches: POST /infer endpoint
    """
    try:
        # If file provided, ingest it first
        if file:
            file_path = Path(file)
            if not file_path.exists():
                typer.echo(f"Error: File not found: {file}", err=True)
                raise typer.Exit(1)
            
            file_ext = file_path.suffix.lower()
            typer.echo(f"📄 Ingesting file: {file_path.name}...")
            
            if file_ext == '.pdf':
                ingest_pdf(str(file_path), title=title)
            elif file_ext == '.txt':
                ingest_text_file(str(file_path), title=title or file_path.stem)
            elif file_ext in ['.png', '.jpg', '.jpeg']:
                ingest_image(str(file_path), title=title or file_path.stem)
            else:
                typer.echo(f"Error: Unsupported file type: {file_ext}. Supported: PDF, TXT, PNG, JPEG", err=True)
                raise typer.Exit(1)
            
            typer.echo(f"✅ Ingested: {file_path.name}")
        
        # Run the query
        typer.echo(f"🔍 Querying: {question}")
        answer = run_deep_rag(question)
        
        typer.echo("\n" + "="*80)
        typer.echo("Answer:")
        typer.echo("="*80)
        typer.echo(answer)
        typer.echo("="*80)
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def health():
    """
    Check if the system is healthy.
    Verifies database connection and basic functionality.
    
    Matches: GET /health endpoint
    """
    try:
        # Try to connect to database
        from retrieval.retrieval import connect
        conn = connect()
        conn.close()
        
        typer.echo("✅ System is healthy")
        typer.echo("  - Database connection: OK")
        return {"ok": True}
    except Exception as e:
        typer.echo(f"❌ System health check failed: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def query_graph(
    question: str = typer.Argument(..., help="The question to ask against ingested documents (uses LangGraph)"),
    thread_id: str = typer.Option("default", "--thread-id", help="Thread ID for conversation state")
):
    """
    Query existing documents using LangGraph pipeline with conditional routing.
    
    The graph allows agents to decide if they have sufficient evidence
    or need to iterate over query refinement and refine_retrieve options.
    
    Matches: POST /ask endpoint (with LangGraph)
    """
    try:
        answer = ask_with_graph(question, thread_id=thread_id)
        typer.echo("\n" + "="*80)
        typer.echo("Answer:")
        typer.echo("="*80)
        typer.echo(answer)
        typer.echo("="*80)
    except Exception as e:
        typer.echo(f"Error querying with graph: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def infer_graph(
    question: str = typer.Argument(..., help="The question to ask"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Optional file to ingest before querying (PDF, TXT, PNG, JPEG)"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Custom title for the document (only used if --file is provided)"),
    thread_id: str = typer.Option("default", "--thread-id", help="Thread ID for conversation state")
):
    """
    Combined ingestion and query using LangGraph pipeline.
    
    If --file is provided:
    - Ingest the file (PDF, TXT, PNG, JPEG)
    - Then query using the question with LangGraph
    
    If no --file:
    - Just query existing documents with LangGraph (same as 'query-graph' command)
    
    Matches: POST /infer endpoint (with LangGraph)
    """
    try:
        # If file provided, ingest it first
        if file:
            file_path = Path(file)
            if not file_path.exists():
                typer.echo(f"Error: File not found: {file}", err=True)
                raise typer.Exit(1)
            
            file_ext = file_path.suffix.lower()
            typer.echo(f"📄 Ingesting file: {file_path.name}...")
            
            if file_ext == '.pdf':
                ingest_pdf(str(file_path), title=title)
            elif file_ext == '.txt':
                ingest_text_file(str(file_path), title=title or file_path.stem)
            elif file_ext in ['.png', '.jpg', '.jpeg']:
                ingest_image(str(file_path), title=title or file_path.stem)
            else:
                typer.echo(f"Error: Unsupported file type: {file_ext}. Supported: PDF, TXT, PNG, JPEG", err=True)
                raise typer.Exit(1)
            
            typer.echo(f"✅ Ingested: {file_path.name}")
        
        # Run the query with LangGraph
        typer.echo(f"🔍 Querying with LangGraph: {question}")
        answer = ask_with_graph(question, thread_id=thread_id)
        
        typer.echo("\n" + "="*80)
        typer.echo("Answer:")
        typer.echo("="*80)
        typer.echo(answer)
        typer.echo("="*80)
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def graph(out: str = typer.Option("deep_rag_graph.png", "--out", "-o", help="Output file path for the graph")):
    """
    Export the LangGraph as PNG (Graphviz) or Mermaid (.mmd fallback).
    """
    try:
        # Lazy import to avoid loading graph_viz unless needed
        from inference.graph.graph_viz import export_graph_png
        path = export_graph_png(out)
        typer.echo(f"✅ Wrote graph to: {path}")
    except Exception as e:
        typer.echo(f"Error generating graph: {e}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
