"""
Infer command - Combined ingestion and query.
"""
import typer
from pathlib import Path
from typing import Optional
from ingestion.ingest import ingest as ingest_pdf
from ingestion.ingest_text import ingest_text_file
from ingestion.ingest_image import ingest_image
from inference.agents import run_deep_rag
from retrieval.retrieval import wait_for_chunks


def infer(
    question: str = typer.Argument(..., help="The question to ask"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Optional file to ingest before querying (PDF, TXT, PNG, JPEG)"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Custom title for the document (only used if --file is provided)"),
    cross_doc: bool = typer.Option(False, "--cross-doc", help="Enable cross-document retrieval (two-stage when doc_id provided)")
):
    """
    Combined ingestion and query endpoint.
    
    If --file is provided:
    - Ingest the file (PDF, TXT, PNG, JPEG)
    - Wait for chunks to be available in the database
    - Then query using the question
    
    If no --file:
    - Just query existing documents (same as 'query' command)
    
    Matches: POST /infer endpoint
    """
    try:
        # If file provided, ingest it first
        doc_id = None
        if file:
            file_path = Path(file)
            if not file_path.exists():
                typer.echo(f"Error: File not found: {file}", err=True)
                raise typer.Exit(1)
            
            file_ext = file_path.suffix.lower()
            typer.echo(f"📄 Ingesting file: {file_path.name}...")
            
            if file_ext == '.pdf':
                doc_id = ingest_pdf(str(file_path), title=title)
            elif file_ext == '.txt':
                doc_id = ingest_text_file(str(file_path), title=title or file_path.stem)
            elif file_ext in ['.png', '.jpg', '.jpeg']:
                doc_id = ingest_image(str(file_path), title=title or file_path.stem)
            else:
                typer.echo(f"Error: Unsupported file type: {file_ext}. Supported: PDF, TXT, PNG, JPEG", err=True)
                raise typer.Exit(1)
            
            if doc_id:
                typer.echo(f"✅ Ingested: {file_path.name}")
                typer.echo(f"📋 Document ID: {doc_id}")
                
                # Wait for chunks to be available before querying
                typer.echo(f"⏳ Waiting for chunks to be available for document {doc_id}...")
                try:
                    chunk_count = wait_for_chunks(doc_id, expected_count=None, max_wait_seconds=30)
                    typer.echo(f"✅ Found {chunk_count} chunks, ready to query")
                    typer.echo(f"🔍 Starting query with document filter: {doc_id}...")
                except TimeoutError as e:
                    typer.echo(f"⚠️  Warning: {e}", err=True)
                    typer.echo("Proceeding with query anyway, but results may be incomplete", err=True)
            else:
                typer.echo(f"⚠️  Warning: Ingestion completed but no document ID returned", err=True)
                raise typer.Exit(1)
        
        # Run the query with doc_id filter if available
        if cross_doc:
            typer.echo("🌐 Cross-document retrieval enabled")
        typer.echo(f"🔍 Querying: {question}")
        answer = run_deep_rag(question, doc_id=doc_id, cross_doc=cross_doc)
        
        typer.echo("\n" + "="*80)
        typer.echo("Answer:")
        typer.echo("="*80)
        typer.echo(answer)
        typer.echo("="*80)
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

