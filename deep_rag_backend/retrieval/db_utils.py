"""
Database utility functions for connecting to PostgreSQL.
Centralized DB connection to avoid DRY violations.
"""
import os
import psycopg2
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


def connect():
    """
    Create and return a PostgreSQL database connection.
    
    Uses environment variables:
    - DB_HOST: Database host (default: localhost)
    - DB_PORT: Database port (default: 5432)
    - DB_USER: Database user
    - DB_PASS: Database password
    - DB_NAME: Database name
    
    Returns:
        psycopg2.connection: Database connection object
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        dbname=os.getenv("DB_NAME")
    )


def get_document_title(doc_id: str) -> Optional[str]:
    """
    Get document title from doc_id.
    
    Args:
        doc_id: Document ID (UUID)
        
    Returns:
        Document title or None if not found
    """
    if not doc_id:
        return None
    
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT title FROM documents WHERE doc_id = %s", (doc_id,))
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        return None
