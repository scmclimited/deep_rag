"""
CLI entry point - Main Typer application.
"""
import typer
from inference.commands import (
    ingest,
    query,
    infer,
    query_graph,
    infer_graph,
    health,
    graph,
    inspect,
    test,
    test_app
)

app = typer.Typer(help="Deep RAG CLI - matches FastAPI service routes")

# Register all commands
app.command()(ingest)
app.command()(query)
app.command()(infer)
app.command()(health)
app.command()(query_graph)
app.command()(infer_graph)
app.command()(graph)
app.command()(inspect)
app.add_typer(test_app, name="test")  # Add test subcommands (test all, test unit, test integration)
app.command()(test)  # Also add as main command for convenience (test [all|unit|integration])

if __name__ == "__main__":
    app()
