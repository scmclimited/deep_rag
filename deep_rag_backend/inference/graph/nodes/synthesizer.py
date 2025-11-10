"""
Synthesizer node: Generates final answer from evidence.
"""
import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypedDict, cast
from inference.graph.state import GraphState
from inference.graph.agent_logger import get_agent_logger
from inference.llm import call_llm
from retrieval.confidence import get_confidence_for_chunks

logger = logging.getLogger(__name__)
agent_log = get_agent_logger()


MAX_CONTEXT_CHUNKS = 8
MAX_CHUNKS_PER_DOC = 2
MAX_KEYWORDS_PER_DOC = 60
MAX_PHRASES_PER_DOC = 30
MAX_LABEL_TERMS = 5

EvidenceChunk = Dict[str, Any]


class DocumentStats(TypedDict):
    score: float
    count: int
    pages: Set[Tuple[Optional[int], Optional[int]]]
    first_index: int


def _chunk_priority_score(chunk: EvidenceChunk) -> float:
    """Blend lexical / vector / cross-encoder scores to rank evidence chunks."""
    lex = float(chunk.get("lex", 0.0))
    vec = float(chunk.get("vec", 0.0))
    ce = float(chunk.get("ce", 0.0))
    # Empirically, ce tends to be the strongest relevance signal, followed by vec.
    return (ce * 0.5) + (vec * 0.35) + (lex * 0.15)


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


def _normalize_for_match(value: str) -> str:
    """Normalize strings so we can line them up against the user question."""
    normalized = re.sub(r"[^a-z0-9 ]+", " ", value.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def select_context_chunks(
    evidence: Sequence[EvidenceChunk],
    selected_doc_ids: Sequence[str],
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    per_doc: int = MAX_CHUNKS_PER_DOC,
) -> List[EvidenceChunk]:
    """
    Diversify the context so each high-value document contributes its strongest chunks.
    """
    if not evidence:
        return []

    doc_chunks: Dict[str, List[Tuple[float, int, EvidenceChunk]]] = defaultdict(list)
    chunks_without_doc: List[Tuple[float, int, EvidenceChunk]] = []

    for idx, ev in enumerate(evidence):
        doc_id = ev.get("doc_id")
        score = _chunk_priority_score(ev)
        if doc_id:
            doc_chunks[doc_id].append((score, idx, ev))
        else:
            chunks_without_doc.append((score, idx, ev))

    if not doc_chunks:
        # No doc IDs present; fall back to the highest scoring chunks overall.
        chunks_without_doc.sort(key=lambda item: (-item[0], item[1]))
        return [ev for _, _, ev in chunks_without_doc[:max_chunks]]

    # Sort chunks inside each document by quality.
    for doc_id in doc_chunks:
        doc_chunks[doc_id].sort(key=lambda item: (-item[0], item[1]))

    # Determine document ordering: respect explicit selection first, then by best chunk score.
    selection_order = [doc for doc in list(selected_doc_ids) if doc in doc_chunks]
    remaining_docs = [
        doc for doc in doc_chunks.keys() if doc not in selection_order
    ]
    remaining_docs.sort(
        key=lambda doc: (-doc_chunks[doc][0][0], doc_chunks[doc][0][1])
    )
    ordered_docs = selection_order + remaining_docs

    context: List[EvidenceChunk] = []

    # Primary pass: take up to `per_doc` chunks per document in order.
    for doc in ordered_docs:
        for score, idx, ev in doc_chunks[doc][:per_doc]:
            context.append(ev)
            if len(context) >= max_chunks:
                return context

    # Secondary pass: fill remaining slots with best leftover chunks regardless of document.
    leftovers: List[Tuple[float, int, EvidenceChunk]] = []
    for doc in ordered_docs:
        leftovers.extend(doc_chunks[doc][per_doc:])
    for doc in doc_chunks:
        if doc not in ordered_docs:
            leftovers.extend(doc_chunks[doc])

    leftovers.sort(key=lambda item: (-item[0], item[1]))
    for _, _, ev in leftovers:
        if len(context) >= max_chunks:
            break
        context.append(ev)

    # As a final fallback, include top chunks without doc IDs if space remains.
    if len(context) < max_chunks and chunks_without_doc:
        chunks_without_doc.sort(key=lambda item: (-item[0], item[1]))
        for _, _, ev in chunks_without_doc:
            if len(context) >= max_chunks:
                break
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
    normalized_question = _normalize_for_match(question_text)
    question_tokens: Set[str] = set(normalized_question.split()) if normalized_question else set()

    doc_stats: Dict[str, DocumentStats] = {}
    doc_keywords: Dict[str, Set[str]] = defaultdict(set)
    doc_phrases: Dict[str, Set[str]] = defaultdict(set)
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

        for alias_key in ("doc_title", "doc_name", "doc_filename", "doc_display", "title", "source_name"):
            alias_value = chunk.get(alias_key)
            if isinstance(alias_value, str) and alias_value.strip():
                doc_aliases[doc_ref].add(alias_value.strip())

        text_value = chunk.get("text")
        if isinstance(text_value, str) and text_value:
            kw_list, phrase_list = _extract_terms(text_value)
            for kw in kw_list:
                if len(doc_keywords[doc_ref]) >= MAX_KEYWORDS_PER_DOC:
                    break
                doc_keywords[doc_ref].add(kw)
            for phrase in phrase_list:
                if len(doc_phrases[doc_ref]) >= MAX_PHRASES_PER_DOC:
                    break
                doc_phrases[doc_ref].add(phrase)

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
    best_score = float(sorted_docs[0][1]["score"]) if sorted_docs else 0.0

    doc_question_positions: Dict[str, Optional[int]] = {}
    doc_question_overlap: Dict[str, int] = {}
    doc_labels: Dict[str, str] = {}

    segment_positions: Dict[str, int] = {}
    segment_overlaps: Dict[str, int] = {}

    segments: List[str] = []
    if question_text:
        # Focus on the portion after a colon if the prompt enumerates documents there.
        segment_source = question_text
        if ":" in question_text:
            segment_source = question_text.split(":", 1)[1]
        for raw_segment in re.split(r",|\band\b", segment_source, flags=re.IGNORECASE):
            segment = raw_segment.strip()
            if len(segment) >= 3:
                segments.append(segment)

    normalized_segments: List[Tuple[int, str, Set[str]]] = []
    for idx, segment in enumerate(segments):
        normalized_segment = _normalize_for_match(segment)
        if not normalized_segment:
            continue
        segment_tokens = set(normalized_segment.split())
        normalized_segments.append((idx, normalized_segment, segment_tokens))

    for doc_ref in score_order:
        keywords = doc_keywords.get(doc_ref, set())
        phrases = doc_phrases.get(doc_ref, set())
        aliases = doc_aliases.get(doc_ref, set())

        keyword_overlap = len(keywords & question_tokens)
        phrase_overlap = _count_phrase_matches(normalized_question, phrases) if normalized_question else 0
        alias_overlap = _count_phrase_matches(normalized_question, aliases) if normalized_question else 0
        doc_question_overlap[doc_ref] = keyword_overlap + phrase_overlap + alias_overlap

        combined_terms: List[str] = list(phrases) + list(keywords) + list(aliases)
        doc_question_positions[doc_ref] = _first_match_position(normalized_question, combined_terms) if normalized_question else None

        doc_token_set: Set[str] = set(keywords)
        for alias in aliases:
            doc_token_set.update(_normalize_for_match(alias).split())
        for phrase in phrases:
            doc_token_set.update(_normalize_for_match(phrase).split())

        for seg_idx, normalized_segment, segment_tokens in normalized_segments:
            overlap = len(doc_token_set & segment_tokens)
            if overlap <= 0:
                continue
            prev_overlap = segment_overlaps.get(doc_ref, 0)
            if overlap > prev_overlap:
                segment_positions[doc_ref] = seg_idx
                segment_overlaps[doc_ref] = overlap

        label_aliases = sorted(aliases, key=lambda item: (len(item), item))
        label = label_aliases[0] if label_aliases else ""
        if not label:
            label_keywords = sorted(keywords, key=lambda item: (len(item), item))
            label = _build_label(label_keywords)
        if not label and phrases:
            label = _build_label(sorted(phrases, key=lambda item: (len(item), item)))
        if not label:
            label = doc_ref[:8]
        doc_labels[doc_ref] = label

    for doc_ref, seg_idx in segment_positions.items():
        if doc_question_positions.get(doc_ref) is None:
            # Multiply to maintain ordering while leaving room for finer-grained positions.
            doc_question_positions[doc_ref] = seg_idx * 1000

    def order_key(doc_ref: str) -> Tuple[int, float, int]:
        position = doc_question_positions.get(doc_ref)
        has_match = 0 if position is not None else 1
        pos_value = float(position) if position is not None else float("inf")
        return (has_match, pos_value, score_order.index(doc_ref))

    top_doc_candidates: List[str] = []
    for doc_ref in score_order:
        stats = doc_stats[doc_ref]
        score_ratio = (stats["score"] / best_score) if best_score > 0 else 0.0
        chunk_count = stats["count"]
        overlap = doc_question_overlap.get(doc_ref, 0)

        include = False
        if doc_ref in explicit_docs:
            include = True
        elif overlap > 0:
            include = True
        elif score_ratio >= 0.55:
            include = True
        elif chunk_count >= 2 and score_ratio >= 0.4:
            include = True

        if include:
            top_doc_candidates.append(doc_ref)
            logger.info(
                "Keeping doc %s (score_ratio=%.2f, chunks=%s, overlap=%s, explicit=%s)",
                doc_ref[:8] + "...",
                score_ratio,
                chunk_count,
                overlap,
                doc_ref in explicit_docs,
            )
        else:
            logger.info(
                "Filtering out doc %s (score_ratio=%.2f, chunks=%s, overlap=%s)",
                doc_ref[:8] + "...",
                score_ratio,
                chunk_count,
                overlap,
            )

    if not top_doc_candidates and score_order:
        logger.info("All documents filtered out; retaining top-scoring document to avoid empty answer")
        top_doc_candidates = [score_order[0]]

    top_doc_ids = sorted(top_doc_candidates, key=order_key)
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

    context_sections: List[str] = []
    if top_doc_ids:
        for doc_ref in top_doc_ids:
            doc_chunks = [chunk for chunk in ctx_evs if chunk.get("doc_id") == doc_ref]
            if not doc_chunks:
                continue
            keywords = sorted(doc_keywords.get(doc_ref, set()), key=lambda item: (len(item), item))
            label = _build_label(keywords) or doc_labels.get(doc_ref, doc_ref[:8])
            snippet = "\n\n".join(str(chunk.get("text", ""))[:1200] for chunk in doc_chunks)
            context_sections.append(f"Document {doc_ref[:8]} (key terms: {label}):\n{snippet}")
        remaining = [chunk for chunk in ctx_evs if chunk.get("doc_id") not in top_doc_ids]
        context_sections.extend(str(chunk.get("text", ""))[:1200] for chunk in remaining)
    else:
        context_sections = [str(chunk.get("text", ""))[:1200] for chunk in ctx_evs]

    context = "\n\n---\n\n".join(context_sections)
    order_block = f"{doc_order_instruction}\n\n" if doc_order_instruction else ""
    instructions = [
        "Answer using ONLY the supplied context.",
        "Do NOT invent documents or details that are not grounded in the context.",
        "Reference each document only if you actually use its information.",
    ]
    if len(doc_order_for_prompt) > 1:
        instructions.append("Organize your answer following the document order listed below.")
        instructions.append("Make it explicit when you transition between documents.")
    base_instructions = "\n".join(f"- {line}" for line in instructions)

    prompt = (
        f"{order_block}{base_instructions}\n\nQuestion: {question_text}\n\nContext:\n{context}\n"
    )

    logger.info("Invoking LLM for synthesis")
    llm_response = call_llm(
        "You write precise, grounded answers. Avoid speculation and keep sources aligned.",
        [{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.2,
    )
    answer_text = llm_response.strip()
    normalized_answer = _normalize_for_match(answer_text)

    negative_response = normalized_answer.startswith("i dont know") or normalized_answer.startswith("i do not know") or normalized_answer.startswith("i don't know")
    if negative_response:
        logger.info("LLM response indicates lack of knowledge - clearing sources")
        cleared_result: Dict[str, Any] = {
            "answer": "I don't know.",
            "confidence": min(overall_confidence, 40.0),
            "action": "abstain",
            "doc_ids": [],
            "pages": [],
        }
        return cast(GraphState, cleared_result)

    answer_tokens: Set[str] = set(normalized_answer.split()) if normalized_answer else set()
    verified_docs: List[str] = []
    for doc_ref in top_doc_ids:
        keywords = doc_keywords.get(doc_ref, set())
        phrases = doc_phrases.get(doc_ref, set())
        aliases = doc_aliases.get(doc_ref, set())
        keyword_matches = len(keywords & answer_tokens)
        phrase_matches = _count_phrase_matches(normalized_answer, phrases) if normalized_answer else 0
        alias_matches = _count_phrase_matches(normalized_answer, aliases) if normalized_answer else 0
        overlap = doc_question_overlap.get(doc_ref, 0)

        stats = doc_stats[doc_ref]
        score_ratio = (stats["score"] / best_score) if best_score > 0 else 0.0

        keep = (keyword_matches > 0) or (phrase_matches > 0) or (alias_matches > 0)

        if keep:
            verified_docs.append(doc_ref)
            logger.info(
                "Verified doc %s (keyword_matches=%s, phrase_matches=%s, alias_matches=%s, score_ratio=%.2f)",
                doc_ref[:8] + "...",
                keyword_matches,
                phrase_matches,
                alias_matches,
                score_ratio,
            )
        else:
            logger.info(
                "Removed doc %s after synthesis (keyword_matches=%s, phrase_matches=%s, alias_matches=%s, overlap=%s)",
                doc_ref[:8] + "...",
                keyword_matches,
                phrase_matches,
                alias_matches,
                overlap,
            )

    if not verified_docs and top_doc_ids:
        fallback_doc = max(top_doc_ids, key=lambda doc: (
            doc_question_overlap.get(doc, 0),
            doc_stats[doc]["score"],
        ))
        logger.info("No documents verified via answer match; falling back to %s", fallback_doc[:8] + "...")
        verified_docs = [fallback_doc]

    if verified_docs:
        verified_docs = sorted(verified_docs, key=order_key)
    else:
        logger.info("No documents verified in final answer - clearing citations")

    top_doc_ids = verified_docs
    citations: List[str] = []
    page_ranges_for_metadata: List[str] = []
    page_numbers: List[int] = []
    if top_doc_ids:
        for idx, doc_ref in enumerate(top_doc_ids, start=1):
            stats = doc_stats[doc_ref]
            sorted_pages = _sort_pages(stats["pages"])
            formatted_pages = [_format_page_range(item) for item in sorted_pages]
            page_str = ", ".join(formatted_pages) if formatted_pages else "p?"
            citations.append(f"[{idx}] doc:{doc_ref} {page_str} (confidence: {overall_confidence:.1f}%)")
            page_ranges_for_metadata.extend(formatted_pages)
            for start_page, _ in sorted_pages:
                if isinstance(start_page, int):
                    page_numbers.append(start_page)

    final_answer = answer_text.rstrip()
    if citations:
        final_answer += "\n\nSources: " + ", ".join(citations)

    primary_doc = doc_id or (top_doc_ids[0] if top_doc_ids else None)
    final_action = "clarify" if action == "clarify" else "answer"
    result_payload: Dict[str, Any] = {
        "answer": final_answer,
        "confidence": overall_confidence,
        "action": final_action,
        "doc_ids": top_doc_ids,
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


def _extract_terms(text: str) -> Tuple[List[str], List[str]]:
    """Extract candidate keywords and phrases from chunk text."""
    normalized = _normalize_for_match(text)
    if not normalized:
        return [], []

    tokens = normalized.split()
    keywords: List[str] = []
    for token in tokens:
        if len(token) >= 3:
            keywords.append(token)

    bigrams: List[str] = []
    for idx in range(len(tokens) - 1):
        first, second = tokens[idx], tokens[idx + 1]
        if len(first) >= 3 and len(second) >= 3:
            bigrams.append(f"{first} {second}")

    trigrams: List[str] = []
    for idx in range(len(tokens) - 2):
        first, second, third = tokens[idx], tokens[idx + 1], tokens[idx + 2]
        if len(first) >= 3 and len(second) >= 3 and len(third) >= 3:
            trigrams.append(f"{first} {second} {third}")

    phrases = bigrams + trigrams
    return keywords[: MAX_KEYWORDS_PER_DOC * 2], phrases[: MAX_PHRASES_PER_DOC * 2]


def _first_match_position(normalized_text: str, candidates: Iterable[str]) -> Optional[int]:
    """Return earliest index where any candidate appears in normalized_text."""
    best_index: Optional[int] = None
    for candidate in candidates:
        candidate_norm = _normalize_for_match(candidate)
        if not candidate_norm:
            continue
        idx = normalized_text.find(candidate_norm)
        if idx == -1:
            continue
        if best_index is None or idx < best_index:
            best_index = idx
    return best_index


def _count_phrase_matches(normalized_text: str, phrases: Iterable[str]) -> int:
    count = 0
    for phrase in phrases:
        normalized_phrase = _normalize_for_match(phrase)
        if normalized_phrase and normalized_phrase in normalized_text:
            count += 1
    return count


def _build_label(keywords: Sequence[str]) -> str:
    if not keywords:
        return ""
    selected = list(keywords)[:MAX_LABEL_TERMS]
    return ", ".join(selected)


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