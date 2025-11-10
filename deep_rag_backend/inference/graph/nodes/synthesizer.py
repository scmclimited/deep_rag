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
    logger.info(
        "State snapshot → iterations=%s, evidence_chunks=%s, action=%s",
        state.get('iterations', 0),
        len(state.get('evidence', []) or []),
        state.get('action'),
    )
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
        # Sort by count (number of chunks) and score
        sorted_docs = sorted(
            doc_stats.items(),
            key=lambda item: (
                -float(item[1]["count"]),
                -float(item[1]["score"]),
                item[1]["first_index"]
            )
        )
        
        # Smart filtering: only include documents that meaningfully contributed
        # Calculate total chunks and score distribution
        if sorted_docs:
            best_count = float(sorted_docs[0][1]["count"])
            best_score = float(sorted_docs[0][1]["score"])
            total_chunks = sum(float(stats["count"]) for _, stats in sorted_docs)
            
            logger.info(f"Document contribution analysis: {len(sorted_docs)} documents, {int(total_chunks)} total chunks")
            for doc, stats in sorted_docs:
                count = float(stats["count"])
                score = float(stats["score"])
                contribution_pct = (count / total_chunks * 100) if total_chunks > 0 else 0
                score_ratio = (score / best_score) if best_score > 0 else 0
                
                logger.info(f"  - {doc[:8]}...: {int(count)} chunks ({contribution_pct:.1f}%), score ratio: {score_ratio:.2f}")
                
                # STRICT filtering: only include documents that are genuinely relevant
                # Include document if it meets ANY of these criteria:
                # 1. Top document AND score ratio > 0.3 (best match with minimum relevance)
                # 2. Has 2+ chunks AND score ratio > 0.7 (multiple chunks + highly relevant)
                # 3. Contributes >40% of chunks AND score ratio > 0.7 (dominant + highly relevant)
                # 4. Score within 90% of best (very highly relevant)
                include = (
                    (doc == sorted_docs[0][0] and score_ratio > 0.3) or  # Top document with minimum relevance
                    (count >= 2 and score_ratio >= 0.7) or  # Multiple chunks + highly relevant
                    (contribution_pct >= 40.0 and score_ratio >= 0.7) or  # Dominant + highly relevant
                    score_ratio >= 0.9  # Very high relevance
                )
                
                if include:
                    top_doc_ids.append(doc)
                    logger.info(f"    ✓ Included: meaningful contribution")
                else:
                    logger.info(f"    ✗ Excluded: insufficient contribution")
                    
        logger.info(f"Final documents for citations: {len(top_doc_ids)} from {len(sorted_docs)} total")
        
        if selection_doc and selection_doc in doc_stats:
            doc_id = selection_doc
            # Ensure selected doc is in top_doc_ids
            if selection_doc not in top_doc_ids:
                top_doc_ids.insert(0, selection_doc)
                logger.info(f"Added explicitly selected document to citations: {doc_id}")
            logger.info(f"Using explicitly selected document: {doc_id}")
        elif not doc_id and top_doc_ids:
            doc_id = top_doc_ids[0]
            logger.info(f"Primary document ID: {doc_id}...")
    
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
        result = {
            "answer": "I don't know.",
            "confidence": overall_confidence,
            "action": "abstain",
            "doc_ids": [],  # Clear doc_ids for abstain
            "pages": []     # Clear pages for abstain
        }
        # Don't include doc_id for abstain responses
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
Do NOT add bracket citations or reference numbers in your response.{doc_context}

Question: {state['question']}

Context:
{context}
"""
    else:
        prompt = f"""Answer the question using ONLY the context provided.

CRITICAL INSTRUCTIONS:
- If insufficient evidence exists, say "I don't know."
- Do NOT add bracket citations or reference numbers in your response - citations will be added automatically.
- Do NOT describe or mention documents that are not directly relevant to answering the question.
- Do NOT fabricate relationships between documents unless explicitly stated in the context.
- Focus ONLY on information that directly answers the question.
- If the context contains multiple documents, only discuss those that are actually relevant to the answer.

Provide a clear, direct answer based on the context.{doc_context}

Question: {state['question']}

Context:
{context}
"""
    ans = call_llm(
        "You write precise, sourced answers. You ONLY discuss information that directly answers the question. You do NOT mention irrelevant documents or fabricate relationships.", 
        [{"role":"user","content":prompt}], 
        max_tokens=1500,  # Increased from 500 to allow for detailed answers with multiple sources
        temperature=0.2
    )
    out = ans.strip()
    normalized_out = out.lower().strip()
    normalized_ascii = normalized_out.encode("ascii", "ignore").decode("ascii") if normalized_out else ""
    
    # Detect "I don't know" or "no relevant documents" responses
    is_negative_response = (
        normalized_out.startswith("i don't know")
        or normalized_out.startswith("i do not know")
        or normalized_ascii.startswith("i dont know")
        or "no other documents" in normalized_out
        or "no documents related" in normalized_out
        or "no relevant documents" in normalized_out
        or "no additional documents" in normalized_out
        or ("there are no" in normalized_out and "document" in normalized_out)
    )
    
    if is_negative_response:
        logger.info(f"Detected negative/no-documents response - clearing sources")
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

