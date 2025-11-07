"""
Synthesizer agent: Generates final answer from evidence.
"""
import logging
from inference.agents.state import State
from inference.llm import call_llm

logger = logging.getLogger(__name__)


def synthesizer(state: State) -> State:
    """Synthesizer agent: Generates final answer from evidence."""
    logger.info("-" * 40)
    logger.info("AGENT: Synthesizer - Generating final answer")
    logger.info("-" * 40)
    logger.info(f"Using top {min(5, len(state['evidence']))} chunks for synthesis")
    
    doc_id = state.get('doc_id')
    if doc_id:
        logger.info(f"Synthesizing answer for specific document: {doc_id}...")
    
    citations = []
    chunks_used = state["evidence"][:5]
    
    # Track all doc_ids from retrieved chunks
    doc_ids_found = set(state.get('doc_ids', []))
    for h in chunks_used:
        hit_doc_id = h.get('doc_id')
        if hit_doc_id:
            doc_ids_found.add(hit_doc_id)
    
    if doc_ids_found:
        state["doc_ids"] = list(doc_ids_found)
        logger.info(f"Identified {len(doc_ids_found)} document(s) from retrieved chunks: {[d + "..." for d in doc_ids_found]}")
        # Use the first doc_id as primary if not already set
        if not doc_id and len(doc_ids_found) == 1:
            doc_id = list(doc_ids_found)[0]
            state["doc_id"] = doc_id
            logger.info(f"Using document ID: {doc_id}...")
    
    # Log which chunks are being used for synthesis
    logger.info("Chunks used for synthesis:")
    for i, h in enumerate(chunks_used, 1):
        chunk_doc_id = h.get('doc_id', 'N/A')
        logger.info(f"  [{i}] Doc: {chunk_doc_id[:8] if chunk_doc_id != 'N/A' else 'N/A'}... Pages {h['p0']}–{h['p1']}: {h.get('text', '')[:100]}...")
    for i, h in enumerate(chunks_used, 1):
        citations.append(f"[{i}] p{h['p0']}–{h['p1']}")
    context = "\n\n".join([f"[{i}] {h['text'][:1200]}" for i, h in enumerate(chunks_used, 1)])
    
    # Include doc_id context in prompt if available
    doc_context = ""
    if doc_id:
        doc_context = f"""\n\nNote: This answer is based on a specific document that was recently ingested or identified from the knowledge base. 
        Document {doc_id} was used for this answer. Focus your answer on this document's content."""
    
    prompt = f"""Answer the question using ONLY the context.
If insufficient evidence, or the result is likely not in the context, say "I don't know."
Add bracket citations like [1], [2] that map to the provided context blocks and snippets of text used from source documents.
Which can include exact verbatim text from source documents or image descriptions.{doc_context}

Question: {state['question']}

Context:
{context}
"""
    ans = call_llm("You write precise, sourced answers.", [{"role":"user","content":prompt}], max_tokens=500)
    state["answer"] = ans.strip() + "\n\nSources: " + ", ".join(citations)
    
    logger.info(f"Generated Answer:\n{state['answer']}")
    logger.info("-" * 40)
    return state

