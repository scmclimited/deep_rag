"""
State definition for LangGraph pipeline.
"""
from typing import TypedDict, List, Dict, Any, Optional


class GraphState(TypedDict, total=False):
    question: str
    plan: str
    evidence: List[Dict[str, Any]]
    notes: str
    answer: str
    confidence: float
    iterations: int
    refinements: List[str]
    doc_id: Optional[str]  # Primary document ID for document-specific retrieval
    doc_ids: List[str]  # All document IDs found during retrieval (for multi-doc tracking)
    cross_doc: bool  # Whether cross-document retrieval is enabled

