"""
Critic agent: Evaluates evidence quality and triggers refinement if needed.
"""
import logging
import re
from inference.agents.state import State
from inference.agents.constants import MAX_ITERS, THRESH
from inference.llm import call_llm
from retrieval.retrieval import retrieve_hybrid

logger = logging.getLogger(__name__)


def critic(state: State) -> State:
    """Critic agent: Evaluates evidence quality and triggers refinement if needed."""
    logger.info("-" * 40)
    logger.info("AGENT: Critic - Evaluating evidence quality")
    logger.info("-" * 40)
    
    ev = state["evidence"]
    strong = sum(1 for h in ev if h.get("ce", 0.0) > THRESH or (h["lex"]>0 and h["vec"]>0))
    conf = min(0.9, 0.4 + 0.1*strong)  # toy heuristic; plug in your own
    state["confidence"] = conf
    
    logger.info(f"Strong chunks: {strong}/{len(ev)}")
    logger.info(f"Confidence score: {conf:.2f}")
    logger.info(f"Iterations: {state['iterations']}/{MAX_ITERS}")
    
    if conf < 0.6 and state["iterations"] < MAX_ITERS:
        logger.info(f"Confidence {conf:.2f} < 0.6 threshold - Requesting refinement...")
        # Ask for refinement: new sub-questions or different keywords
        prompt = f"""Given the plan:\n{state['plan']}\nAnd notes:\n{state['notes']}\n
Propose refined sub-queries (max 2) to retrieve missing evidence. Short, 1 line each.

IMPORTANT: Write queries as natural language questions without special characters like &, *, |, !, :, or quotes. 
Use plain text only. For example, write "Hygiene and DX" instead of "Hygiene & DX"."""
        refinements = call_llm("You suggest refinements for the given question and plan.", [{"role":"user","content":prompt}], max_tokens=120)
        # Re-query once with the first refinement
        rq_raw = refinements.splitlines()[0].strip("-• ").strip()
        # Sanitize the refinement query
        rq = rq_raw.replace('&', ' and ')
        rq = re.sub(r'[\!\|\:\*\"]', ' ', rq)
        rq = re.sub(r'\s+', ' ', rq).strip()
        logger.info(f"Refinement query: {rq}")
        
        doc_id = state.get('doc_id')
        cross_doc = state.get('cross_doc', False)
        hits = retrieve_hybrid(rq, k=6, k_lex=30, k_vec=30, doc_id=doc_id, cross_doc=cross_doc)
        
        # Track doc_ids from refinement retrieval
        doc_ids_found = set(state.get('doc_ids', []))
        for hit in hits:
            hit_doc_id = hit.get('doc_id')
            if hit_doc_id:
                doc_ids_found.add(hit_doc_id)
        state["doc_ids"] = list(doc_ids_found)
        logger.info(f"Retrieved {len(hits)} additional chunks from refinement")
        
        # Merge and dedup by chunk_id
        seen, merged = set(), []
        for h in state["evidence"] + hits:
            if h["chunk_id"] in seen: continue
            seen.add(h["chunk_id"]); merged.append(h)
        state["evidence"] = merged
        state["iterations"] += 1
        
        logger.info(f"Total evidence after merge: {len(state['evidence'])} chunks")
        logger.info("-" * 80)
        return critic(state)   # one self-loop step (bounded by MAX_ITERS)
    else:
        if conf >= 0.6:
            logger.info(f"Confidence {conf:.2f} >= 0.6 - Proceeding to synthesis")
        else:
            logger.warning(f"Max iterations ({MAX_ITERS}) reached with confidence {conf:.2f}")
    logger.info("-" * 80)
    return state

