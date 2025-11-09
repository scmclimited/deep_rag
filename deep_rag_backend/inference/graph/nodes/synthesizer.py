"""
Synthesizer node: Generates final answer from evidence.
"""
import logging
import unicodedata
from typing import Dict, List
from inference.graph.state import GraphState
from inference.graph.agent_logger import get_agent_logger
from inference.llm import call_llm
from retrieval.confidence import get_confidence_for_chunks

logger = logging.getLogger(__name__)
agent_log = get_agent_logger()


def node_synthesizer(state: GraphState) -> GraphState:
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Synthesizer - Generating final answer")
    logger.info("-" * 40)
    
    # Handle None evidence properly
    evidence = state.get("evidence")
    if evidence is None:
        evidence = []
    
    logger.info(f"Using top {min(5, len(evidence))} chunks for synthesis")
    
    doc_id = state.get('doc_id')
    if doc_id:
        logger.info(f"Synthesizing answer for specific document: {doc_id}...")
    
    ctx_evs = evidence[:5] if evidence else []
    doc_stats: Dict[str, Dict[str, object]] = {}
    doc_order: List[str] = []
    selection_doc = None
    selected_doc_ids = state.get('selected_doc_ids')
    if isinstance(selected_doc_ids, list):
        for candidate in selected_doc_ids:
            if candidate:
                selection_doc = candidate
                break
    if not selection_doc and doc_id:
        selection_doc = doc_id

    for idx, ev in enumerate(ctx_evs):
        ev_doc_id = ev.get('doc_id')
        if not ev_doc_id:
            continue
        if ev_doc_id not in doc_stats:
            doc_stats[ev_doc_id] = {
                "score": 0.0,
                "count": 0,
                "pages": set(),
                "first_index": idx
            }
            doc_order.append(ev_doc_id)
        score = (ev.get('lex', 0.0) * 0.6) + (ev.get('vec', 0.0) * 0.4)
        doc_stats[ev_doc_id]["score"] = float(doc_stats[ev_doc_id]["score"]) + score
        doc_stats[ev_doc_id]["count"] = int(doc_stats[ev_doc_id]["count"]) + 1
        p0 = ev.get('p0')
        p1 = ev.get('p1')
        if p0 is not None:
            doc_stats[ev_doc_id]["pages"].add((p0, p1))
    
    # If no evidence/chunks retrieved, always abstain
    if not ctx_evs or len(ctx_evs) == 0:
        logger.info("No evidence retrieved - abstaining")
        result = {"answer": "I don't know.", "confidence": 0.0, "action": "abstain"}
        if doc_id:
            result["doc_id"] = doc_id
        return result
    
    top_doc_ids: List[str] = []
    if doc_stats:
        sorted_docs = sorted(
            doc_stats.items(),
            key=lambda item: (
                -float(item[1]["count"]),
                -float(item[1]["score"]),
                item[1]["first_index"]
            )
        )
        if sorted_docs:
            best_score = float(sorted_docs[0][1]["score"])
            top_doc_ids = [
                doc for doc, stats in sorted_docs
                if abs(float(stats["score"]) - best_score) < 1e-6
            ]
        logger.info(f"Top document(s) selected for synthesis: {top_doc_ids}")
        if selection_doc and selection_doc in doc_stats:
            doc_id = selection_doc
            top_doc_ids = [selection_doc]
            logger.info(f"Using explicitly selected document: {doc_id}")
        elif not doc_id and top_doc_ids:
            doc_id = top_doc_ids[0]
            logger.info(f"Using document ID: {doc_id}...")
    
    # Calculate overall confidence using multi-feature approach
    question = state.get('question', '')
    conf_result = get_confidence_for_chunks(ctx_evs, query=question)
    overall_confidence = conf_result["confidence"]
    overall_probability = conf_result["probability"]
    action = conf_result["action"]
    
    logger.info(f"Confidence: {overall_confidence:.2f}% (probability: {overall_probability:.3f}), Action: {action}, Thresholds: abstain<{conf_result.get('abstain_threshold', 0.45)*100:.1f}%, clarify<{conf_result.get('clarify_threshold', 0.65)*100:.1f}%")
    
    # Handle abstain action - also check if confidence is very low (< 40%) even if above threshold
    # This provides an extra safety check for cases with no documents
    if action == "abstain" or overall_confidence < 40.0:
        result = {"answer": "I don't know.", "confidence": overall_confidence, "action": "abstain"}
        if doc_id:
            result["doc_id"] = doc_id
        logger.info(f"Abstaining due to low confidence ({overall_confidence:.2f}%)")
        return result
    
    # Build citations with overall confidence score (not per-chunk)
    # Use the overall confidence that was calculated for the entire answer
    citations = []
    overall_confidence_pct = f"{overall_confidence:.1f}%"
    
    def format_page_range(pages_tuple):
        p0, p1 = pages_tuple
        if p0 is None:
            return "p?"
        if p1 is not None and p1 != p0:
            return f"p{p0}-{p1}"
        return f"p{p0}"

    allowed_docs = set(top_doc_ids) if top_doc_ids else None
    ordered_docs = top_doc_ids or doc_order
    for idx, doc in enumerate(ordered_docs, start=1):
        if allowed_docs and doc not in allowed_docs:
            continue
        stats = doc_stats.get(doc)
        if not stats:
            continue
        pages = sorted(stats["pages"])
        if not pages:
            continue
        formatted_pages = sorted({format_page_range(pg) for pg in pages})
        page_str = ", ".join(formatted_pages)
        citations.append(f"[{idx}] doc:{doc} {page_str} (confidence: {overall_confidence_pct})")
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
    ans = call_llm("You write precise, sourced answers.", [{"role":"user","content":prompt}], max_tokens=500, temperature=0.2)
    out = ans.strip()
    normalized_out = out.lower().strip()
    normalized_ascii = normalized_out.encode("ascii", "ignore").decode("ascii") if normalized_out else ""
    if (
        normalized_out.startswith("i don't know")
        or normalized_out.startswith("i do not know")
        or normalized_ascii.startswith("i dont know")
    ):
        citations = []
        top_doc_ids = []
        doc_id = None
        action = "abstain"
        overall_confidence = min(overall_confidence, 40.0)
    if citations:
        out += "\n\nSources: " + ", ".join(citations)
    
    # Update state with doc_ids and confidence
    result = {
        "answer": out,
        "confidence": overall_confidence,
        "action": action
    }
    if top_doc_ids:
        result["doc_ids"] = top_doc_ids
        logger.info(f"Answer generated for {len(top_doc_ids)} document(s): {[d + '...' for d in top_doc_ids]}")
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
        confidence=overall_confidence,
        iterations=state.get('iterations', 0),
        metadata={
            "citations": citations,
            "answer_length": len(out),
            "doc_id": doc_id if doc_id else None,
            "doc_ids": top_doc_ids,
            "confidence_action": action,
            "confidence_features": conf_result.get("features", {})
        }
    )
    
    return result

