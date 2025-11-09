"""
Synthesizer agent: Generates final answer from evidence.
"""
import logging
from inference.agents.state import State
from inference.llm import call_llm
from retrieval.confidence import get_confidence_for_chunks

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
    
    chunks_used = state["evidence"][:5]
    
    # If no evidence/chunks retrieved, always abstain
    if not chunks_used or len(chunks_used) == 0:
        logger.info("No evidence retrieved - abstaining")
        state["answer"] = "I don't know."
        state["confidence"] = 0.0
        return state
    
    # Track all doc_ids from retrieved chunks
    doc_ids_found = set(state.get('doc_ids', []))
    for h in chunks_used:
        hit_doc_id = h.get('doc_id')
        if hit_doc_id:
            doc_ids_found.add(hit_doc_id)
    
    if doc_ids_found:
        state["doc_ids"] = list(doc_ids_found)
        logger.info(f"Identified {len(doc_ids_found)} document(s) from retrieved chunks: {[d + '...' for d in doc_ids_found]}")
        # Use the first doc_id as primary if not already set
        if not doc_id and len(doc_ids_found) == 1:
            doc_id = list(doc_ids_found)[0]
            state["doc_id"] = doc_id
            logger.info(f"Using document ID: {doc_id}...")
    
    # Calculate overall confidence using multi-feature approach
    question = state.get('question', '')
    conf_result = get_confidence_for_chunks(chunks_used, query=question)
    overall_confidence = conf_result["confidence"]
    overall_probability = conf_result["probability"]
    action = conf_result["action"]
    
    logger.info(f"Confidence: {overall_confidence:.2f}% (probability: {overall_probability:.3f}), Action: {action}, Thresholds: abstain<{conf_result.get('abstain_threshold', 0.45)*100:.1f}%, clarify<{conf_result.get('clarify_threshold', 0.65)*100:.1f}%")
    
    # Handle abstain action - also check if confidence is very low (< 40%) even if above threshold
    # This provides an extra safety check for cases with no documents
    if action == "abstain" or overall_confidence < 40.0:
        state["answer"] = "I don't know."
        state["confidence"] = overall_confidence
        logger.info(f"Abstaining due to low confidence ({overall_confidence:.2f}%)")
        return state
    
    # Log which chunks are being used for synthesis
    logger.info("Chunks used for synthesis:")
    for i, h in enumerate(chunks_used, 1):
        chunk_doc_id = h.get('doc_id', 'N/A')
        logger.info(f"  [{i}] Doc: {chunk_doc_id[:8] if chunk_doc_id != 'N/A' else 'N/A'}... Pages {h['p0']}–{h['p1']}: {h.get('text', '')[:100]}...")
    
    # Build citations with per-chunk confidence scores
    citations = []
    for i, h in enumerate(chunks_used, 1):
        chunk_doc_id = h.get('doc_id')
        
        # Calculate per-chunk confidence (simpler approach for citations)
        lex_score = float(h.get('lex', 0.0) or 0.0)
        vec_score = float(h.get('vec', 0.0) or 0.0)
        ce_score = float(h.get('ce', 0.0) or 0.0)
        
        # Weighted combination for per-chunk display
        if ce_score > 0:
            chunk_confidence = (0.2 * lex_score + 0.3 * vec_score + 0.5 * ce_score) * 100
        else:
            chunk_confidence = (0.4 * lex_score + 0.6 * vec_score) * 100
        
        confidence_pct = f"{chunk_confidence:.1f}%"
        
        if chunk_doc_id:
            citations.append(f"[{i}] doc:{chunk_doc_id} p{h['p0']}–{h['p1']} (confidence: {confidence_pct})")
        else:
            citations.append(f"[{i}] p{h['p0']}–{h['p1']} (confidence: {confidence_pct})")
    context = "\n\n".join([f"[{i}] {h['text'][:1200]}" for i, h in enumerate(chunks_used, 1)])
    
    # Include doc_id context in prompt if available
    doc_context = ""
    if doc_id:
        doc_context = f"""\n\nNote: This answer is based on a specific document that was recently ingested or identified from the knowledge base. 
        Document {doc_id} was used for this answer. Focus your answer on this document's content."""
    
    # Adjust prompt based on action (clarify vs answer)
    if action == "clarify":
        prompt = f"""Using ONLY the context, summarize cautiously in 1–2 sentences.
If the answer is incomplete, say what's missing.
Add bracket citations like [1], [2] that map to the provided context blocks.{doc_context}

Question: {state['question']}

Context:
{context}
"""
    else:
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
    state["confidence"] = overall_confidence
    
    logger.info(f"Generated Answer:\n{state['answer']}")
    logger.info("-" * 40)
    return state

