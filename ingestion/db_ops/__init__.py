"""
Database operations package for ingestion.
"""
from ingestion.db_ops.document import upsert_document
from ingestion.db_ops.chunks import upsert_chunks

__all__ = [
    "upsert_document",
    "upsert_chunks",
]

