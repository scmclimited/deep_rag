"""
Critic node: Evaluates evidence quality and triggers refinement if needed.
"""
import logging
import re
from inference.graph.state import GraphState
from inference.graph.constants import MAX_ITERS, THRESH
from inference.graph.agent_logger import get_agent_logger
from inference.llm import call_llm

logger = logging.getLogger(__name__)
agent_log = get_agent_logger()


def node_critic(state: GraphState) -> GraphState:
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Critic - Evaluating evidence quality")
    logger.info(f"State snapshot → iterations={state.get('iterations', 0)}, evidence_chunks={len(state.get('evidence', []))}")
    logger.info("-" * 40)
    
    ev = state.get("evidence", [])
    strong = sum(1 for h in ev if float(h.get("ce", 0.0)) > THRESH or (h.get("lex",0)>0 and h.get("vec",0)>0))
    conf = min(0.9, 0.4 + 0.1*strong)

    result: GraphState = {"confidence": conf, "iterations": state.get("iterations", 0)}

    logger.info(f"Strong chunks: {strong}/{len(ev)}")
    logger.info(f"Confidence score: {conf:.2f}")
    logger.info(f"Iterations: {state.get('iterations', 0)}/{MAX_ITERS}")
    
    # Log critic evaluation
    agent_log.log_step(
        node="critic",
        action="evaluate",
        confidence=conf,
        iterations=state.get('iterations', 0),
        metadata={
            "strong_chunks": strong,
            "total_chunks": len(ev),
            "threshold": THRESH
        }
    )

    # If weak confidence and not at loop cap, propose refinements (sub-queries)
    if conf < 0.6 and state.get("iterations", 0) < MAX_ITERS:
        logger.info(f"Confidence {conf:.2f} < 0.6 threshold - Requesting refinement...")
        logger.info(f"Current iteration: {state.get('iterations', 0)}/{MAX_ITERS}")
        
        # Enhanced prompt for multi-document queries
        question = state.get('question', '')
        is_multi_doc_query = any(keyword in question.lower() for keyword in [
            'all documents', 'these documents', 'multiple documents', 'each document',
            'contents of', 'share the contents', 'what documents'
        ])
        
        if is_multi_doc_query:
            logger.info("Detected multi-document query - using enhanced refinement strategy")
            prompt = f"""Given the plan:\n{state.get('plan','')}\nAnd notes:\n{state.get('notes','')}\n

This is a multi-document query. The user wants comprehensive information from multiple documents.
Propose refined sub-queries (max 2) to retrieve MORE complete evidence from the documents.
Focus on:
1. Retrieving more chunks from each document
2. Getting document metadata (titles, types, structure)
3. Extracting key content sections

Write queries as natural language questions without special characters like &, *, |, !, :, or quotes. 
Use plain text only. For example, write "Hygiene and DX" instead of "Hygiene & DX"."""
        else:
            prompt = f"""Given the plan:\n{state.get('plan','')}\nAnd notes:\n{state.get('notes','')}\n
Propose refined sub-queries (max 2) to retrieve missing evidence. Short, 1 line each.

IMPORTANT: Write queries as natural language questions without special characters like &, *, |, !, :, or quotes. 
Use plain text only. For example, write "Hygiene and DX" instead of "Hygiene & DX"."""
        
        refinements = call_llm("You suggest refinements.", [{"role":"user","content":prompt}], max_tokens=120, temperature=0.0)
        lines = [ln.strip("-• ").strip() for ln in refinements.splitlines() if ln.strip()]
        # Additional sanitization: remove any remaining special characters
        sanitized_lines = []
        for line in lines:
            # Remove leading special chars and normalize
            cleaned = line.strip()
            # Replace & with "and" if present
            cleaned = cleaned.replace('&', ' and ')
            # Remove other problematic characters
            cleaned = re.sub(r'[\!\|\:\*\"]', ' ', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned:
                sanitized_lines.append(cleaned)
        
        result["refinements"] = sanitized_lines[:2] if sanitized_lines else []
        result["iterations"] = state.get("iterations", 0) + 1
        
        logger.info(f"Generated {len(result['refinements'])} refinement(s):")
        for i, ref in enumerate(result['refinements'], 1):
            logger.info(f"  {i}. {ref}")
        logger.info(f"Next iteration will be: {result['iterations']}/{MAX_ITERS}")
        logger.info("Routing to refine_retrieve node")
        
        # Log refinement decision
        agent_log.log_step(
            node="critic",
            action="request_refinement",
            confidence=conf,
            iterations=result["iterations"],
            refinements=result["refinements"]
        )
    else:
        result["refinements"] = []
        if conf >= 0.6:
            logger.info(f"Confidence {conf:.2f} >= 0.6 - Routing to synthesizer")
        else:
            logger.warning(
                "Max iterations (%s) reached. Critic heuristic confidence (0-1 scale): %.2f. "
                "The synthesizer will compute final user-facing confidence (percentage) next.",
                MAX_ITERS,
                conf,
            )
    logger.info("-" * 40)
    return result

