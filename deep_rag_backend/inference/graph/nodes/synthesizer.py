"""
Synthesizer node: Generates final answer from evidence.
"""
import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypedDict, cast
from inference.graph.state import GraphState
from inference.graph.agent_logger import get_agent_logger
from inference.llm import call_llm
from retrieval.confidence import get_confidence_for_chunks
from retrieval.db_utils import get_document_title

logger = logging.getLogger(__name__)
agent_log = get_agent_logger()


MAX_CONTEXT_CHUNKS = 24  # Increased to allow more context for verbose documents
MAX_CHUNKS_PER_DOC = 6  # Increased from 2 to allow more chunks per document for long/verbose docs

EvidenceChunk = Dict[str, Any]


class DocumentStats(TypedDict):
    score: float
    count: int
    pages: Set[Tuple[Optional[int], Optional[int]]]
    first_index: int


def _normalize_doc_ids(value: Any) -> List[str]:
    if isinstance(value, list):
        result: List[str] = []
        for item in cast(List[Any], value):
            if item is not None and item != "":
                result.append(str(item))
        return result
    if value is not None and value != "":
        return [str(value)]
    return []


def select_context_chunks(
    evidence: Sequence[EvidenceChunk],
    selected_doc_ids: Sequence[str],
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    per_doc: int = MAX_CHUNKS_PER_DOC,
) -> List[EvidenceChunk]:
    """
    Select context chunks while preserving retrieval order.
    Respects explicit document selection and applies per-doc limits.
    """
    if not evidence:
        return []

    # Preserve retrieval order - chunks are already ranked by relevance from retriever
    # Track chunks per document to apply per-doc limit
    doc_chunk_counts: Dict[str, int] = defaultdict(int)
    
    context: List[EvidenceChunk] = []
    chunks_without_doc: List[EvidenceChunk] = []

    # First pass: process chunks in retrieval order, applying per-doc limits
    for ev in evidence:
        if len(context) >= max_chunks:
            break
            
        doc_id = ev.get("doc_id")
        
        if not doc_id:
            # Chunks without doc_id go to separate list
            chunks_without_doc.append(ev)
            continue
        
        # Apply per-doc limit
        if doc_chunk_counts[doc_id] >= per_doc:
            continue  # Skip - already have enough chunks from this doc
        
        # Add chunk and increment counter
        context.append(ev)
        doc_chunk_counts[doc_id] += 1

    # Second pass: fill remaining slots with other chunks (preserving retrieval order)
    if len(context) < max_chunks:
        for ev in evidence:
            if len(context) >= max_chunks:
                break
                
            doc_id = ev.get("doc_id")
            
            # Skip if already in context
            if ev in context:
                continue
            
            if not doc_id:
                # Add chunks without doc_id
                context.append(ev)
                continue
            
            # Apply per-doc limit
            if doc_chunk_counts[doc_id] >= per_doc:
                continue
            
            context.append(ev)
            doc_chunk_counts[doc_id] += 1

    # Final pass: add chunks without doc_id if space remains
    if len(context) < max_chunks and chunks_without_doc:
        for ev in chunks_without_doc:
            if len(context) >= max_chunks:
                break
            if ev not in context:
                context.append(ev)

    return context


def node_synthesizer(state: GraphState) -> GraphState:
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Synthesizer - Generating final answer")
    logger.info(
        "State snapshot → iterations=%s, evidence_chunks=%s, action=%s",
        state.get("iterations", 0),
        len(state.get("evidence", []) or []),
        state.get("action"),
    )
    logger.info("-" * 40)

    evidence = state.get("evidence") or []
    logger.info(f"Total chunks retrieved: {len(evidence)}")
    for idx, chunk in enumerate(evidence):
        doc_ref = chunk.get("doc_id")
        preview = str(chunk.get("text", ""))[:80].replace("\n", " ")
        logger.info(f"  Chunk {idx}: doc={doc_ref if doc_ref else 'None'} preview={preview}...")

    selected_doc_ids = _normalize_doc_ids(state.get("selected_doc_ids"))
    uploaded_doc_ids = _normalize_doc_ids(state.get("uploaded_doc_ids"))
    explicit_docs = set(selected_doc_ids + uploaded_doc_ids)

    raw_doc_id = state.get("doc_id")
    doc_id = str(raw_doc_id) if raw_doc_id else None
    if doc_id:
        logger.info(f"Primary document requested: {doc_id}")

    question_text = state.get("question", "") or ""

    doc_stats: Dict[str, DocumentStats] = {}
    doc_aliases: Dict[str, Set[str]] = defaultdict(set)
    doc_order: List[str] = []

    for idx, chunk in enumerate(evidence):
        doc_ref = chunk.get("doc_id")
        if not isinstance(doc_ref, str) or not doc_ref:
            continue

        if doc_ref not in doc_stats:
            doc_stats[doc_ref] = DocumentStats(score=0.0, count=0, pages=set(), first_index=idx)
            doc_order.append(doc_ref)

        stats = doc_stats[doc_ref]
        stats["count"] += 1
        score = (float(chunk.get("lex", 0.0)) * 0.6) + (float(chunk.get("vec", 0.0)) * 0.4)
        stats["score"] += score

        p0 = chunk.get("p0")
        p1 = chunk.get("p1")
        if isinstance(p0, int) and isinstance(p1, int):
            stats["pages"].add((p0, p1))
        elif isinstance(p0, int):
            stats["pages"].add((p0, None))

        # Collect document aliases (titles, names, etc.) for labels
        for alias_key in ("doc_title", "doc_name", "doc_filename", "doc_display", "title", "source_name"):
            alias_value = chunk.get(alias_key)
            if isinstance(alias_value, str) and alias_value.strip():
                doc_aliases[doc_ref].add(alias_value.strip())

    logger.info(f"Document stats collected: {len(doc_stats)} document(s)")
    for doc_ref, stats in doc_stats.items():
        logger.info(
            "  Doc %s: count=%s, score=%.4f, pages=%s",
            doc_ref[:8] + "...",
            stats["count"],
            stats["score"],
            len(stats["pages"]),
        )

    ctx_evs = select_context_chunks(evidence, selected_doc_ids)
    if not ctx_evs:
        logger.info("No context chunks available - abstaining")
        abstain_result: Dict[str, Any] = {
            "answer": "I don't know.",
            "confidence": 0.0,
            "action": "abstain",
            "doc_ids": [],
            "pages": [],
        }
        return cast(GraphState, abstain_result)

    if not doc_stats:
        logger.info("Context available but no document statistics - treating as single anonymous document")
        anonymous_result: Dict[str, Any] = {
            "answer": "I don't know.",
            "confidence": 0.0,
            "action": "abstain",
            "doc_ids": [],
            "pages": [],
        }
        return cast(GraphState, anonymous_result)

    sorted_docs = sorted(
        doc_stats.items(),
        key=lambda item: (
            -float(item[1]["count"]),
            -float(item[1]["score"]),
            item[1]["first_index"],
        ),
    )
    score_order = [doc for doc, _ in sorted_docs]

    # Simplified: Build labels for context sections (LLM sees these)
    # Use document aliases (titles) if available, otherwise use doc_id prefix
    doc_labels: Dict[str, str] = {}
    for doc_ref in score_order:
        aliases = doc_aliases.get(doc_ref, set())
        label_aliases = sorted(aliases, key=lambda item: (len(item), item))
        label = label_aliases[0] if label_aliases else doc_ref[:8]
        doc_labels[doc_ref] = label

    # Simplified document selection: prioritize explicit docs, then use score order
    # No complex filtering - let the LLM decide what to use
    top_doc_candidates: List[str] = []
    
    # First, include all explicit docs (user-selected or uploaded)
    for doc_ref in score_order:
        if doc_ref in explicit_docs:
            top_doc_candidates.append(doc_ref)
            logger.info(f"Including explicit doc {doc_ref[:8]}...")
    
    # Then add top-scoring docs that aren't already included
    for doc_ref in score_order:
        if doc_ref not in top_doc_candidates:
            top_doc_candidates.append(doc_ref)
            if len(top_doc_candidates) >= 10:  # Reasonable limit
                break
    
    if not top_doc_candidates and score_order:
        logger.info("No documents selected; using top-scoring document")
        top_doc_candidates = [score_order[0]]

    top_doc_ids = top_doc_candidates
    doc_order_for_prompt = top_doc_ids if top_doc_ids else score_order

    ordered_chunks: List[EvidenceChunk] = []
    if top_doc_ids:
        doc_set = set(top_doc_ids)
        for doc_ref in top_doc_ids:
            ordered_chunks.extend([chunk for chunk in ctx_evs if chunk.get("doc_id") == doc_ref])
        ordered_chunks.extend([chunk for chunk in ctx_evs if chunk.get("doc_id") not in doc_set])
    else:
        ordered_chunks = list(ctx_evs)
    ctx_evs = ordered_chunks

    doc_order_lines: List[str] = []
    for idx, doc_ref in enumerate(doc_order_for_prompt, start=1):
        label = doc_labels.get(doc_ref, doc_ref[:8])
        doc_order_lines.append(f"{idx}. key terms: {label}")
    doc_order_instruction = "Document order for your response:\n" + "\n".join(doc_order_lines) if doc_order_lines else ""

    question_for_confidence = state.get("question", "") or ""
    conf_result = get_confidence_for_chunks(ctx_evs, query=question_for_confidence)
    overall_confidence = conf_result["confidence"]
    overall_probability = conf_result["probability"]
    action = conf_result["action"]
    logger.info(
        "Confidence %.2f%% (probability=%.3f) action=%s",
        overall_confidence,
        overall_probability,
        action,
    )

    if action == "abstain" or overall_confidence < 40.0:
        logger.info("Confidence too low - abstaining")
        abstain_result: Dict[str, Any] = {
            "answer": "I don't know.",
            "confidence": overall_confidence,
            "action": "abstain",
            "doc_ids": [],
            "pages": [],
        }
        return cast(GraphState, abstain_result)

    # Build document reference map with titles for citation format
    doc_ref_map: List[Dict[str, str]] = []
    for idx, doc_ref in enumerate(top_doc_ids, start=1):
        doc_title = get_document_title(doc_ref)
        doc_ref_map.append({
            "number": str(idx),
            "doc_id": doc_ref,
            "doc_prefix": doc_ref[:8],
            "title": doc_title or f"Document {doc_ref[:8]}",
            "citation_format": f"[DOC {doc_ref[:8]}]"
        })
    
    # Build document reference list for LLM prompt
    doc_reference_list = ""
    if doc_ref_map:
        doc_reference_list = "\n\nAvailable Documents (use the citation format when referencing):\n"
        for doc_info in doc_ref_map:
            doc_reference_list += f"- Document {doc_info['number']}: {doc_info['title']} (Reference: {doc_info['citation_format']})\n"
        doc_reference_list += "\nWhen you reference information from a document, use the format [DOC {doc[:8]}] where {doc[:8]} is the first 8 characters of the document ID shown above.\n"
        doc_reference_list += "Example: If discussing content from Document 1 with ID prefix 'a1b2c3d4', write: According to [DOC a1b2c3d4], the information shows...\n"

    context_sections: List[str] = []
    if top_doc_ids:
        for doc_ref in top_doc_ids:
            doc_chunks = [chunk for chunk in ctx_evs if chunk.get("doc_id") == doc_ref]
            if not doc_chunks:
                continue
            label = doc_labels.get(doc_ref, doc_ref[:8])
            snippet = "\n\n".join(str(chunk.get("text", ""))[:1200] for chunk in doc_chunks)
            context_sections.append(f"Document {doc_ref[:8]} ({label}):\n{snippet}")
        remaining = [chunk for chunk in ctx_evs if chunk.get("doc_id") not in top_doc_ids]
        context_sections.extend(str(chunk.get("text", ""))[:1200] for chunk in remaining)
    else:
        context_sections = [str(chunk.get("text", ""))[:1200] for chunk in ctx_evs]

    context = "\n\n---\n\n".join(context_sections)
    order_block = f"{doc_order_instruction}\n\n" if doc_order_instruction else ""

    doc_context = ""
    if doc_id:
        doc_context = "\n\nNote: Focus your answer on the identified document."

    question_lower = question_text.lower()
    is_content_request = any(
        phrase in question_lower
        for phrase in [
            "share the contents",
            "what is in",
            "what are in",
            "contents of",
            "summarize these",
            "tell me about these",
            "describe these",
        ]
    )
    is_multi_doc_query = len(selected_doc_ids) > 1

    if action == "clarify":
        prompt = f"""{order_block}{doc_reference_list}Using ONLY the context, summarize cautiously in 1–3 sentences.\nIf the answer is incomplete, say what's missing.\nWhen referencing documents, use the format [DOC {{doc[:8]}}] as shown in the document reference list above.{doc_context}\n\nQuestion: {question_text}\n\nContext:\n{context}\n"""
    elif is_content_request and is_multi_doc_query:
        prompt = f"""{doc_reference_list}You are analyzing {len(selected_doc_ids)} documents. Provide a comprehensive summary of each document's key information.\n\n
        CRITICAL INSTRUCTIONS:\n- Extract and present the main content, key points, and important details from EACH document no matter how small the detail may seem.\n-
        You must reply and cite documents in the order they are mentioned to you, match the ask/request/question, or likely better match the flow of the context and the question or request prose.
        You must make it explicit when you transition between documents by using the citation format [DOC {{doc[:8]}}] when referencing each document.
        You must include specific information like names, dates, numbers, and key facts.
        You must use proper nouns and pronouns correctly.
        You must be thorough and detailed - the user wants comprehensive information about ALL documents.
        You must NOT say you cannot share contents - you CAN and SHOULD summarize the key information.
        You must if the context lacks information needed to answer any portion of the request, reply exactly with "I don't know." and nothing else.
        When referencing a document, you MUST use the format [DOC {{doc[:8]}}] as shown in the document reference list above.
        Organize your answer by document and use the citation format when transitioning between documents.\n- 
        Follow the document order listed below exactly\n- 
        Be thorough and detailed - the user wants comprehensive information about ALL documents\n- 
        Do NOT say you cannot share contents - you CAN and SHOULD summarize the key information\n- 
        If the context lacks information needed to answer any portion of the request, reply exactly with "I don't know." and nothing else\n- 
        Use the citation format [DOC {{doc[:8]}}] when referencing documents in your answer\n\n{order_block}Question: {question_text}{doc_context}\n\nContext from {len(selected_doc_ids)} documents:\n{context}\n"""
    else:
        prompt = f"""{doc_reference_list}Answer the question using ONLY the context provided.\n\n
        CRITICAL INSTRUCTIONS:\n- If insufficient evidence exists, say "I don't know."\n-
        You must include specific information like names, dates, numbers, and key facts.
        You must use proper nouns and pronouns correctly.\n- 
        Your refusal must be exactly "I don't know." with no extra text when the answer cannot be determined from the context.\n- 
        When referencing information from a document, you MUST not forget to format and cite the document with the citation format [DOC {{doc[:8]}}] as shown in the document reference list above.\n- 
        Do NOT describe or mention documents that are not directly relevant to answering the question.\n- 
        Do NOT fabricate relationships between documents unless explicitly stated in the context.\n- 
        Focus ONLY on information that directly answers the question.\n-
        You do not exist as an entity, you are a helpful asssistant who is only there to extract information or described what is presented given the posed question or request: {question_text}.  
        If the context contains multiple documents, only discuss those that are actually relevant to the answer and use the citation format when referencing them.\n- 
        Follow the document order listed (if provided) when structuring your answer.\n\n{order_block}Provide a clear, direct answer based on the context. When you reference information from a document, use the format [DOC {{doc[:8]}}] as shown above.{doc_context}\n\nQuestion: {question_text}\n\nContext:\n{context}\n"""

    logger.info("Invoking LLM for synthesis")
    llm_response = call_llm(
        "You write precise, grounded answers. Avoid speculation and keep sources aligned. Answer with I dont know if you cannot ground your answer.",
        [{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.2,
    )
    answer_text = llm_response.strip()
    
    # Build citations for all top_doc_ids (pruning will happen in citation_pruner)
    citations: List[str] = []
    page_ranges_for_metadata: List[str] = []
    page_numbers: List[int] = []
    if top_doc_ids:
        # Track pages from chunks that were in context
        doc_pages: Dict[str, Set[Tuple[Optional[int], Optional[int]]]] = defaultdict(set)
        for chunk in ctx_evs:
            chunk_doc_id = chunk.get("doc_id")
            if chunk_doc_id and chunk_doc_id in top_doc_ids:
                p0 = chunk.get("p0")
                p1 = chunk.get("p1")
                if isinstance(p0, int) and isinstance(p1, int):
                    doc_pages[chunk_doc_id].add((p0, p1))
                elif isinstance(p0, int):
                    doc_pages[chunk_doc_id].add((p0, None))
        
        for idx, doc_ref in enumerate(top_doc_ids, start=1):
            # Use pages from context chunks
            pages_from_context = doc_pages.get(doc_ref, set())
            if not pages_from_context:
                # Fallback to stats pages if no context pages found
                pages_from_context = doc_stats[doc_ref]["pages"]
            
            sorted_pages = _sort_pages(pages_from_context)
            formatted_pages = [_format_page_range(item) for item in sorted_pages]
            page_str = ", ".join(formatted_pages) if formatted_pages else "p?"
            citations.append(f"[{idx}] doc:{doc_ref} {page_str} (confidence: {overall_confidence:.1f}%)")
            page_ranges_for_metadata.extend(formatted_pages)
            for start_page, _ in sorted_pages:
                if isinstance(start_page, int):
                    page_numbers.append(start_page)

    final_answer = answer_text.rstrip()
    # Don't add Sources section here - citation_pruner will handle it

    primary_doc = doc_id or (top_doc_ids[0] if top_doc_ids else None)
    final_action = "clarify" if action == "clarify" else "answer"
    
    # Pass through all top_doc_ids - citation_pruner will prune based on LLM answer
    result_payload: Dict[str, Any] = {
        "answer": final_answer,
        "confidence": overall_confidence,
        "action": final_action,
        "doc_ids": top_doc_ids,  # Pass all documents - pruner will filter
        "pages": sorted(set(page_numbers)),
    }
    if primary_doc:
        result_payload["doc_id"] = primary_doc
    if citations:
        result_payload["citations"] = citations

    logger.info(f"Generated answer for {len(top_doc_ids)} document(s)")
    logger.info(final_answer)
    logger.info("-" * 40)

    agent_log.log_step(
        node="synthesizer",
        action="synthesize",
        question=question_text,
        answer=final_answer,
        num_chunks=len(ctx_evs),
        pages=sorted(set(page_numbers)) if page_numbers else None,
        confidence=overall_confidence,
        iterations=state.get("iterations", 0),
        metadata={
            "doc_ids": top_doc_ids,
            "citations": citations,
            "page_ranges": page_ranges_for_metadata,
            "confidence_action": final_action,
            "confidence_features": conf_result.get("features", {}),
        },
    )

    return cast(GraphState, result_payload)


def _sort_pages(pages: Iterable[Tuple[Optional[int], Optional[int]]]) -> List[Tuple[Optional[int], Optional[int]]]:
    def sort_key(item: Tuple[Optional[int], Optional[int]]) -> Tuple[int, int]:
        start = item[0] if item[0] is not None else 10**6
        end = item[1] if item[1] is not None else start
        return (start, end)

    return sorted(pages, key=sort_key)


def _format_page_range(page_range: Tuple[Optional[int], Optional[int]]) -> str:
    start, end = page_range
    if start is None:
        return "p?"
    if end is not None and end != start:
        return f"p{start}-{end}"
    return f"p{start}"