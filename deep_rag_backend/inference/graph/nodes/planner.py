"""
Planner node: Decomposes the question into sub-goals.
"""
import logging
from inference.graph.state import GraphState
from inference.graph.agent_logger import get_agent_logger
from inference.llm import call_llm

logger = logging.getLogger(__name__)
agent_log = get_agent_logger()


def node_planner(state: GraphState) -> GraphState:
    logger.info("=" * 80)
    logger.info("GRAPH NODE: Planner - Decomposing question into sub-goals")
    logger.info("=" * 80)
    logger.info(f"State snapshot:")
    logger.info(f"  - Iterations: {state.get('iterations', 0)}")
    logger.info(f"  - Cross-doc: {state.get('cross_doc', False)}")
    logger.info(f"  - Selected doc IDs: {state.get('selected_doc_ids')}")
    logger.info(f"  - Doc ID: {state.get('doc_id')}")
    logger.info("-" * 80)
    logger.info(f"Question: {state['question']}")
    doc_id = state.get('doc_id')
    selected_doc_ids = state.get('selected_doc_ids')
    if selected_doc_ids and len(selected_doc_ids) > 0:
        logger.info(f"Planning for {len(selected_doc_ids)} selected document(s): {[d[:8] + '...' for d in selected_doc_ids]}")
    elif doc_id:
        logger.info(f"Planning for specific document: {doc_id[:8]}...")
    
    # Include doc_id context in prompt if available
    doc_context = ""
    if doc_id:
        doc_context = f"""\n\nNote: This question is about a specific document that was just ingested. 
        Document {doc_id} was used for this planning. Focus your planning on this document's content."""
    
    prompt = f"""You are a planner. Decompose the user's question into 1-3 concrete sub-goals
that can be answered ONLY from the provided assets such as PDFs, images, or other documents. Prefer explicit nouns and constraints.
Question: {state['question']}{doc_context}"""
    plan = call_llm("You plan tasks.", [{"role": "user", "content": prompt}], max_tokens=200, temperature=0.2)
    plan_text = plan.strip()
    
    logger.info(f"Generated Plan: {plan_text}")
    logger.info("-" * 40)
    
    # Log to agent logger for future training
    agent_log.log_step(
        node="planner",
        action="plan_generation",
        question=state['question'],
        plan=plan_text,
        iterations=state.get("iterations", 0)
    )
    
    return {"plan": plan_text, "iterations": state.get("iterations", 0)}

