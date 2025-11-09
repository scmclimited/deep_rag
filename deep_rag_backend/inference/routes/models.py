"""
Pydantic models for REST API requests.
"""
from pydantic import BaseModel
from typing import Optional


class AskBody(BaseModel):
    question: str
    doc_id: Optional[str] = None  # Optional document ID to filter retrieval to a specific document
    cross_doc: bool = False  # Enable cross-document retrieval (two-stage when doc_id provided)


class InferBody(BaseModel):
    question: str
    title: Optional[str] = None


class AskGraphBody(BaseModel):
    question: str
    thread_id: Optional[str] = "default"
    user_id: Optional[str] = None  # User ID for thread tracking
    doc_id: Optional[str] = None  # Optional document ID to filter retrieval to a specific document
    selected_doc_ids: Optional[list[str]] = None  # Optional list of document IDs for multi-document selection (not cross-doc)
    cross_doc: bool = False  # Enable cross-document retrieval (two-stage when doc_id provided)

