"""
Document database operations.
"""
import hashlib
import os
from uuid import uuid4
from typing import Optional


def calculate_content_hash(file_path: str) -> str:
    """
    Calculate SHA256 hash of file content for duplicate detection.
    
    Args:
        file_path: Path to file
        
    Returns:
        SHA256 hash as hex string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def upsert_document(cur, title: str, path: str, content_hash: Optional[str] = None) -> str:
    """
    Insert a new document record, checking for duplicates by content hash and title.
    
    Args:
        cur: Database cursor
        title: Document title
        path: Document file path
        content_hash: Optional content hash (SHA256). If not provided, will be calculated.
        
    Returns:
        Document ID (UUID string) - existing doc_id if duplicate found, new doc_id otherwise
    """
    # Calculate content hash if not provided
    if content_hash is None and os.path.exists(path):
        content_hash = calculate_content_hash(path)
    
    # Check for duplicate: same content hash AND same title
    if content_hash:
        cur.execute(
            """
            SELECT doc_id FROM documents 
            WHERE content_hash = %s AND title = %s
            LIMIT 1
            """,
            (content_hash, title)
        )
        existing = cur.fetchone()
        if existing:
            # Duplicate found - return existing doc_id
            return existing[0]
    
    # No duplicate found - insert new document
    did = uuid4()
    cur.execute(
        "INSERT INTO documents (doc_id, title, source_path, content_hash) VALUES (%s,%s,%s,%s)",
        (str(did), title, path, content_hash)
    )
    return str(did)

