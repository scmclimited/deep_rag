"""
Health command - Check system health.
"""
import typer
from retrieval.db_utils import connect


def health():
    """
    Check if the system is healthy.
    Verifies database connection and basic functionality.
    
    Matches: GET /health endpoint
    """
    try:
        # Try to connect to database
        conn = connect()
        conn.close()
        
        typer.echo("✅ System is healthy")
        typer.echo("  - Database connection: OK")
        return {"ok": True}
    except Exception as e:
        typer.echo(f"❌ System health check failed: {e}", err=True)
        raise typer.Exit(1)

