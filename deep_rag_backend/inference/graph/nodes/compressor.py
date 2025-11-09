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
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Compressor - Summarizing evidence")
    logger.info("-" * 40)
    logger.info(f"Compressing {len(state.get('evidence', []))} chunks into notes...")
    
    snippets = "\n\n".join([f"[p{h['p0']}–{h['p1']}] {h['text'][:1200]}" for h in state.get("evidence", [])])
    prompt = f"""Summarize the following snippets into crisp notes with bullets.
Retain numbers and proper nouns verbatim. Avoid speculation.
Snippets:\n{snippets}"""
    notes = call_llm("You compress evidence.", [{"role": "user", "content": prompt}], max_tokens=400, temperature=0.1)
    notes_text = notes.strip()
    
    logger.info(f"Compressed Notes:\n{notes_text}")
    logger.info("-" * 40)
    
    # Log compression step
    agent_log.log_step(
        node="compressor",
        action="compress",
        num_chunks=len(state.get('evidence', [])),
        metadata={"notes_length": len(notes_text)}
    )
    
    return {"notes": notes_text}

