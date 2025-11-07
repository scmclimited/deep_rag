"""
Compressor agent: Summarizes retrieved evidence into concise notes.
"""
import logging
from inference.agents.state import State
from inference.llm import call_llm

logger = logging.getLogger(__name__)


def compressor(state: State) -> State:
    """Compressor agent: Summarizes retrieved evidence into concise notes."""
    logger.info("-" * 40)
    logger.info("AGENT: Compressor - Summarizing evidence")
    logger.info("-" * 40)
    logger.info(f"Compressing {len(state['evidence'])} chunks into notes...")
    
    # Map-reduce style compression of top evidence
    snippets = "\n\n".join([f"[p{h['p0']}–{h['p1']}] {h['text'][:1200]}" for h in state["evidence"]])
    prompt = f"""Summarize the following context into crisp notes with bullets.
Retain numbers and proper nouns verbatim. Avoid speculation.
Context:\n{snippets}"""
    notes = call_llm("You compress evidence from grounded context.", [{"role":"user","content":prompt}], max_tokens=300)
    state["notes"] = notes.strip()
    
    logger.info(f"Compressed Notes:\n{state['notes']}")
    logger.info("-" * 80)
    return state

