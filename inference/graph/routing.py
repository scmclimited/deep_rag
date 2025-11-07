"""
Conditional routing functions for LangGraph.
"""
from inference.graph.state import GraphState
from inference.graph.constants import MAX_ITERS


def should_refine(state: GraphState) -> str:
    """Edge function: either loop to refine_retrieve or end at synthesizer."""
    if state.get("confidence", 0.0) < 0.6 and state.get("iterations", 0) <= MAX_ITERS and state.get("refinements"):
        return "refine"
    return "synthesize"

