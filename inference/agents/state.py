"""
State definition for direct agent pipeline.
"""
from typing import TypedDict, List, Optional


class State(TypedDict, total=False):
    question: str
    plan: str
    evidence: List[dict]
    notes: str
    answer: str
    confidence: float
    iterations: int
    doc_id: Optional[str]  # Primary document ID for document-specific retrieval
    doc_ids: List[str]  # All document IDs found during retrieval (for multi-doc tracking)
    cross_doc: bool  # Whether cross-document retrieval is enabled

