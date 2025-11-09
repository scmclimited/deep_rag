"""
Planner agent: Decomposes the question into sub-goals.
"""
import logging
from inference.agents.state import State
from inference.llm import call_llm

logger = logging.getLogger(__name__)


def planner(state: State) -> State:
    """Planner agent: Decomposes the question into sub-goals."""
    logger.info("-" * 40)
    logger.info("AGENT: Planner - Decomposing question into sub-goals")
    logger.info("-" * 40)
    logger.info(f"Question: {state['question']}")
    doc_id = state.get('doc_id')
    if doc_id:
        logger.info(f"Planning for specific document: {doc_id}...")
    
    # Include doc_id context in prompt if available
    doc_context = ""
    if doc_id:
        doc_context = f"\n\nNote: This question is about a specific document that was just ingested. Focus your planning on this document's content."
    
    prompt = f"""You are a planner. Decompose the user's question into 1-3 concrete sub-goals
that can be answered ONLY from the provided context. Prefer explicit nouns and constraints.
Question: {state['question']}{doc_context}"""
    plan = call_llm("You plan tasks for the given question.", [{"role":"user","content":prompt}], max_tokens=350)
    state["plan"] = plan.strip()
    
    logger.info(f"Generated Plan: {state['plan']}")
    logger.info("-" * 80)
    return state

