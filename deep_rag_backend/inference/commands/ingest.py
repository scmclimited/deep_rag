"""
Ingest command - Ingest documents without querying.
"""
import typer
from pathlib import Path
from typing import Optional
from ingestion.ingest import ingest as ingest_pdf
from ingestion.ingest_text import ingest_text_file
from ingestion.ingest_image import ingest_image


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
        doc_id = None
        if file_ext == '.pdf':
            doc_id = ingest_pdf(str(file_path), title=title)
            typer.echo(f"✅ Ingested PDF: {file_path.name}")
        elif file_ext == '.txt':
            doc_id = ingest_text_file(str(file_path), title=title or file_path.stem)
            typer.echo(f"✅ Ingested text file: {file_path.name}")
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            doc_id = ingest_image(str(file_path), title=title or file_path.stem)
            typer.echo(f"✅ Ingested image: {file_path.name}")
        else:
            typer.echo(f"Error: Unsupported file type: {file_ext}. Supported: PDF, TXT, PNG, JPEG", err=True)
            raise typer.Exit(1)
        
        if doc_id:
            typer.echo(f"📋 Document ID: {doc_id}")
        else:
            typer.echo(f"⚠️  Warning: Ingestion completed but no document ID returned", err=True)
    except Exception as e:
        typer.echo(f"Error ingesting file: {e}", err=True)
        raise typer.Exit(1)

