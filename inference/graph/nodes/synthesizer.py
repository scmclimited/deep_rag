"""
Synthesizer node: Generates final answer from evidence.
"""
import logging
from inference.graph.state import GraphState
from inference.graph.agent_logger import get_agent_logger
from inference.llm import call_llm

logger = logging.getLogger(__name__)
agent_log = get_agent_logger()


def node_synthesizer(state: GraphState) -> GraphState:
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Synthesizer - Generating final answer")
    logger.info("-" * 40)
    logger.info(f"Using top {min(5, len(state.get('evidence', [])))} chunks for synthesis")
    
    doc_id = state.get('doc_id')
    if doc_id:
        logger.info(f"Synthesizing answer for specific document: {doc_id}...")
    
    ctx_evs = state.get("evidence", [])[:5]
    
    # Identify doc_ids from retrieved chunks if not already set
    if not doc_id and ctx_evs:
        doc_ids_found = set(h.get('doc_id') for h in ctx_evs if h.get('doc_id'))
        if doc_ids_found:
            logger.info(f"Identified {len(doc_ids_found)} document(s) from retrieved chunks: {[d + '...' for d in doc_ids_found]}")
            # Use the most common doc_id if multiple found
            if len(doc_ids_found) == 1:
                doc_id = list(doc_ids_found)[0]
                logger.info(f"Using document ID: {doc_id}...")
    
    # Build citations with doc_id if available
    citations = []
    for i, h in enumerate(ctx_evs, 1):
        chunk_doc_id = h.get('doc_id')
        if chunk_doc_id:
            # Include full doc_id in citation: [1] doc:c60b6642-d489-4fff-aba7-f146c32862d8 p1-1
            citations.append(f"[{i}] doc:{chunk_doc_id} p{h['p0']}–{h['p1']}")
        else:
            # Fallback to page-only citation if no doc_id
            citations.append(f"[{i}] p{h['p0']}–{h['p1']}")
    # Log which chunks are being used for synthesis
    logger.info("Chunks used for synthesis:")
    for i, h in enumerate(ctx_evs, 1):
        chunk_doc_id = h.get('doc_id', 'N/A')
        logger.info(f"  [{i}] Doc: {chunk_doc_id[:8] if chunk_doc_id != 'N/A' else 'N/A'}... Pages {h['p0']}–{h['p1']}: {h.get('text', '')[:100]}...")
    context = "\n\n".join([f"[{i}] {h['text'][:1200]}" for i, h in enumerate(ctx_evs, 1)])
    
    # Include doc_id context in prompt if available
    doc_context = ""
    if doc_id:
        doc_context = f"\n\nNote: This answer is based on a specific document that was recently ingested or identified from the knowledge base. Focus your answer on this document's content."
    
    prompt = f"""Answer the question using ONLY the context.
If insufficient evidence, or the result is likely not in the context, say "I don't know."
Add bracket citations like [1], [2] that map to the provided context blocks and snippets of text used from source documents.
Which can include exact verbatim text from source documents or image descriptions.{doc_context}

Question: {state['question']}

Context:
{context}
"""
    ans = call_llm("You write precise, sourced answers.", [{"role":"user","content":prompt}], max_tokens=500, temperature=0.2)
    out = ans.strip()
    if citations:
        out += "\n\nSources: " + ", ".join(citations)
    
    # Update state with doc_ids
    result = {"answer": out}
    doc_ids = state.get('doc_ids', [])
    if doc_ids:
        result["doc_ids"] = doc_ids
        logger.info(f"Answer generated for {len(doc_ids)} document(s): {[d + '...' for d in doc_ids]}")
    elif doc_id:
        result["doc_id"] = doc_id
        logger.info(f"Answer generated for document: {doc_id}...")
    
    logger.info(f"Generated Answer:\n{out}")
    logger.info("-" * 40)
    
    # Log final synthesis
    agent_log.log_step(
        node="synthesizer",
        action="synthesize",
        question=state['question'],
        answer=out,
        num_chunks=len(ctx_evs),
        pages=sorted(set([h['p0'] for h in ctx_evs])),
        confidence=state.get('confidence', 0.0),
        iterations=state.get('iterations', 0),
        metadata={
            "citations": citations,
            "answer_length": len(out),
            "doc_id": doc_id if doc_id else None
        }
    )
    
    return result

