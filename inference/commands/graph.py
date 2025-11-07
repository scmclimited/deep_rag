"""
Graph command - Export LangGraph visualization.
"""
import typer
from inference.graph.graph_viz import export_graph_png


def graph(out: str = typer.Option("deep_rag_graph.png", "--out", "-o", help="Output file path for the graph")):
    """
    Export the LangGraph as PNG (Graphviz) or Mermaid (.mmd fallback).
    """
    try:
        # Lazy import to avoid loading graph_viz unless needed
        path = export_graph_png(out)
        typer.echo(f"✅ Wrote graph to: {path}")
    except Exception as e:
        typer.echo(f"Error generating graph: {e}", err=True)
        raise typer.Exit(1)

