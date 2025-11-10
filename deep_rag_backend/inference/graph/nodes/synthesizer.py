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
        # CRITICAL: For multi-doc attachments, disable strict filtering
        # User explicitly attached N documents and wants info about ALL of them
        selected_doc_ids = state.get('selected_doc_ids', []) or []
        is_multi_doc_attachment = isinstance(selected_doc_ids, list) and len(selected_doc_ids) > 1
        
        if sorted_docs:
            best_count = float(sorted_docs[0][1]["count"])
            best_score = float(sorted_docs[0][1]["score"])
            total_chunks = sum(float(stats["count"]) for _, stats in sorted_docs)
            
            logger.info(f"Document contribution analysis: {len(sorted_docs)} documents, {int(total_chunks)} total chunks")
            if is_multi_doc_attachment:
                logger.info(f"📎 Multi-document attachment detected ({len(selected_doc_ids)} docs) - including ALL attached documents in citations")
            
            for doc, stats in sorted_docs:
                count = float(stats["count"])
                score = float(stats["score"])
                contribution_pct = (count / total_chunks * 100) if total_chunks > 0 else 0
                score_ratio = (score / best_score) if best_score > 0 else 0
                
                logger.info(f"  - {doc[:8]}...: {int(count)} chunks ({contribution_pct:.1f}%), score ratio: {score_ratio:.2f}")
                
                # For multi-doc attachments: include ALL documents that were explicitly attached
                # For cross-doc search: apply strict filtering
                if is_multi_doc_attachment and doc in selected_doc_ids:
                    # User explicitly attached this document - always include it
                    include = True
                    logger.info(f"    ✓ Included: explicitly attached document")
                else:
                    # RELAXED filtering: include documents that contribute meaningfully
                    # Include document if it meets ANY of these criteria:
                    # 1. Top document AND score ratio > 0.2 (best match with any relevance)
                    # 2. Has 2+ chunks AND score ratio > 0.5 (multiple chunks + moderate relevance)
                    # 3. Contributes >20% of chunks AND score ratio > 0.5 (significant contribution)
                    # 4. Score within 60% of best (good relevance)
                    # 5. Has at least 1 chunk AND score ratio > 0.4 (any contribution with decent relevance)
                    include = (
                        (doc == sorted_docs[0][0] and score_ratio > 0.2) or  # Top document with any relevance
                        (count >= 2 and score_ratio >= 0.5) or  # Multiple chunks + moderate relevance
                        (contribution_pct >= 20.0 and score_ratio >= 0.5) or  # Significant contribution
                        score_ratio >= 0.6 or  # Good relevance
                        (count >= 1 and score_ratio >= 0.4)  # Any contribution with decent relevance
                    )
                    
                    if include:
                        logger.info(f"    ✓ Included: meaningful contribution")
                    else:
                        logger.info(f"    ✗ Excluded: insufficient contribution")
                
                if include:
                    top_doc_ids.append(doc)
                    
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
    
    # Build context WITHOUT bracket notation to prevent LLM from using [1], [2], etc. in responses
    # The LLM should describe documents naturally, not reference chunk indices
    context = "\n\n---\n\n".join([h['text'][:1200] for h in ctx_evs])
    
    # Include doc_id context in prompt if available
    doc_context = ""
    if doc_id:
        doc_context = f"\n\nNote: This answer is based on a specific document that was recently ingested or identified from the knowledge base. Focus your answer on this document's content."
    
    # Detect if this is a multi-document content request
    question_lower = state.get('question', '').lower()
    is_content_request = any(phrase in question_lower for phrase in [
        'share the contents', 'what is in', 'what are in', 'contents of', 
        'summarize these', 'tell me about these', 'describe these'
    ])
    is_multi_doc_query = selected_doc_ids and len(selected_doc_ids) > 1
    
    # Adjust prompt based on action and query type
    if action == "clarify":
        prompt = f"""Using ONLY the context, summarize cautiously in 1–2 sentences.
If the answer is incomplete, say what's missing.
Do NOT add bracket citations or reference numbers in your response.{doc_context}

Question: {state['question']}

Context:
{context}
"""
    elif is_content_request and is_multi_doc_query:
        # Special prompt for multi-document content requests
        prompt = f"""You are analyzing {len(selected_doc_ids)} documents. Provide a comprehensive summary of each document's key information.

CRITICAL INSTRUCTIONS:
- Extract and present the main content, key points, and important details from EACH document
- Organize your answer by document (e.g., "Document 1 contains...", "Document 2 discusses...")
- Include specific information like names, dates, numbers, and key facts
- Be thorough and detailed - the user wants comprehensive information about ALL documents
- Do NOT say you cannot share contents - you CAN and SHOULD summarize the key information
- Do NOT add bracket citations or reference numbers - citations will be added automatically

Question: {state['question']}

Context from {len(selected_doc_ids)} documents:
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
    # HYBRID APPROACH: Two-pass synthesis for multi-document queries
    # Pass 1: Generate answer
    logger.info("Pass 1: Generating answer from evidence")
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
    else:
        # HYBRID APPROACH: Pass 2 - Post-processing citation alignment
        # For multi-document ATTACHMENT queries (not cross-doc), verify which documents were actually mentioned
        # Skip for cross-doc queries as they are exploratory and should show all found documents
        cross_doc = state.get('cross_doc', False)
        if len(top_doc_ids) > 1 and not cross_doc:
            logger.info(f"Pass 2: Post-processing citation alignment for {len(top_doc_ids)} documents")
            
            # Build a mapping of doc_id to document title/identifier for verification
            doc_id_to_title = {}
            for doc in top_doc_ids:
                # Extract a short identifier for each document
                doc_id_to_title[doc] = doc[:8]  # Use first 8 chars as identifier
            
            # Ask LLM to identify which documents were actually used in the answer
            verification_prompt = f"""Given the following answer and list of available documents, identify which documents were ACTUALLY used to generate the answer.

Answer:
{out}

Available Documents:
{chr(10).join([f"- Document {doc[:8]}..." for doc in top_doc_ids])}

Instructions:
- List ONLY the document IDs that were directly referenced or used in the answer
- If a document was not mentioned or used, do NOT include it
- Format: One document ID per line (just the 8-char prefix)
- If NO documents were used (e.g., "I don't know" response), respond with "NONE"

Which documents were used?"""
            
            used_docs_response = call_llm(
                "You identify which documents were used in an answer.",
                [{"role": "user", "content": verification_prompt}],
                max_tokens=200,
                temperature=0.0
            )
            
            used_doc_ids = []
            if used_docs_response.strip().upper() != "NONE":
                # Parse the response to extract document IDs
                for line in used_docs_response.strip().splitlines():
                    line = line.strip().strip('-').strip()
                    if line:
                        # Find matching doc_id
                        for doc in top_doc_ids:
                            if doc.startswith(line) or line in doc:
                                if doc not in used_doc_ids:
                                    used_doc_ids.append(doc)
                                break
            
            logger.info(f"Citation alignment: {len(used_doc_ids)}/{len(top_doc_ids)} documents actually used")
            logger.info(f"  Original: {[d[:8] + '...' for d in top_doc_ids]}")
            logger.info(f"  Verified: {[d[:8] + '...' for d in used_doc_ids]}")
            
            # Update top_doc_ids to only include verified documents
            if used_doc_ids:
                top_doc_ids = used_doc_ids
                # Rebuild citations with only verified documents
                citations = []
                for idx, doc in enumerate(top_doc_ids, start=1):
                    stats = doc_stats.get(doc)
                    if not stats:
                        continue
                    pages = sorted(stats["pages"])
                    if not pages:
                        continue
                    formatted_pages = sorted({format_page_range(pg) for pg in pages})
                    page_str = ", ".join(formatted_pages)
                    citations.append(f"[{idx}] doc:{doc} {page_str} (confidence: {overall_confidence_pct})")
                logger.info(f"Updated citations: {citations}")
            else:
                # No documents were actually used - clear citations
                logger.info("No documents were verified as used - clearing citations")
                citations = []
                top_doc_ids = []
                doc_id = None
        else:
            if len(top_doc_ids) > 1:
                logger.info(f"Cross-doc query with {len(top_doc_ids)} documents - skipping Pass 2 verification")
            else:
                logger.info("Single document query - skipping citation alignment pass")
    
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

