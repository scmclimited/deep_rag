"""
Database utility functions for connecting to PostgreSQL.
Centralized DB connection to avoid DRY violations.
Uses connection pooling for high-concurrency workloads.
"""
import os
import logging
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
from typing import Optional
from contextlib import contextmanager

load_dotenv()

logger = logging.getLogger(__name__)

# Global connection pool (initialized on first use)
_connection_pool: Optional[pool.ThreadedConnectionPool] = None


def _get_pool():
    """
    Get or create the global connection pool.
    
    Uses ThreadedConnectionPool for thread-safe connection management.
    Pool size: 10-50 connections (adjust based on UVICORN_WORKERS * expected concurrent queries)
    """
    global _connection_pool
    if _connection_pool is None:
        try:
            logger.info("Initializing PostgreSQL connection pool (2-30 connections)")
            _connection_pool = pool.ThreadedConnectionPool(
                minconn=2,   # Start with minimal connections
                maxconn=30,  # Maximum connections (should be < POSTGRES_MAX_CONNECTIONS)
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASS"),
                dbname=os.getenv("DB_NAME")
            )
            logger.info("PostgreSQL connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    return _connection_pool


@contextmanager
def connect():
    """
    Context manager that returns a connection from the pool.
    
    Usage:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")
    
    The connection is automatically returned to the pool when the context exits.
    Falls back to direct connection if pool initialization fails.
    """
    try:
        conn_pool = _get_pool()
        conn = conn_pool.getconn()
        try:
            yield conn
        finally:
            conn_pool.putconn(conn)
    except Exception as e:
        logger.warning(f"Connection pool unavailable, using direct connection: {e}")
        # Fallback to direct connection (old behavior)
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            dbname=os.getenv("DB_NAME")
        )
        try:
            yield conn
        finally:
            conn.close()


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
