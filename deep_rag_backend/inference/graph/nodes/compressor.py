"""
Compressor node: Summarizes retrieved evidence into concise notes.
"""
import logging
from inference.graph.state import GraphState
from inference.graph.agent_logger import get_agent_logger
from inference.llm import call_llm

logger = logging.getLogger(__name__)
agent_log = get_agent_logger()


def node_compressor(state: GraphState) -> GraphState:
    logger.info("=" * 80)
    logger.info("GRAPH NODE: Compressor - Summarizing evidence")
    logger.info("=" * 80)
    logger.info(f"State snapshot:")
    logger.info(f"  - Iterations: {state.get('iterations', 0)}")
    logger.info(f"  - Evidence chunks: {len(state.get('evidence', []))}")
    logger.info(f"  - Cross-doc: {state.get('cross_doc', False)}")
    logger.info("-" * 80)
    
    evidence = state.get("evidence", [])
    logger.info(f"Compressing {len(evidence)} chunks into notes...")
    
    # Log document distribution in evidence
    doc_distribution = {}
    for h in evidence:
        doc_id = h.get('doc_id', 'unknown')
        doc_distribution[doc_id] = doc_distribution.get(doc_id, 0) + 1
    
    if doc_distribution:
        logger.info(f"Evidence distribution across documents:")
        for doc_id, count in sorted(doc_distribution.items(), key=lambda x: -x[1]):
            logger.info(f"  - {doc_id[:8]}...: {count} chunk(s)")
    
    snippets = "\n\n".join([f"[p{h['p0']}–{h['p1']}] {h['text'][:1200]}" for h in evidence])
    prompt = f"""Summarize the following snippets into crisp notes with bullets.
Retain numbers and proper nouns verbatim. Avoid speculation.
Snippets:\n{snippets}"""
    notes = call_llm("You compress evidence.", [{"role": "user", "content": prompt}], max_tokens=400, temperature=0.1)
    notes_text = notes.strip()
    
    logger.info(f"Compressed Notes (length: {len(notes_text)} chars):")
    logger.info(f"{notes_text[:500]}..." if len(notes_text) > 500 else notes_text)
    logger.info("-" * 80)
    
    # Log compression step
    agent_log.log_step(
        node="compressor",
        action="compress",
        num_chunks=len(state.get('evidence', [])),
        metadata={"notes_length": len(notes_text)}
    )
    
    return {"notes": notes_text}

