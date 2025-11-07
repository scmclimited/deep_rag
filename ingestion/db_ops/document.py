"""
Document database operations.
"""
from uuid import uuid4


def upsert_document(cur, title: str, path: str) -> str:
    """
    Insert a new document record.
    
    Args:
        cur: Database cursor
        title: Document title
        path: Document file path
        
    Returns:
        Document ID (UUID string)
    """
    did = uuid4()
    cur.execute(
        "INSERT INTO documents (doc_id, title, source_path) VALUES (%s,%s,%s)",
        (str(did), title, path)
    )
    return str(did)

