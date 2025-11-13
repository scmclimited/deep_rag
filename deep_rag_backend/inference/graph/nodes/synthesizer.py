"""
Synthesizer node: Generates final answer from evidence.
"""
import os
from dotenv import load_dotenv
from collections import defaultdict
import logging
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    cast,
)

from inference.graph.agent_logger import get_agent_logger
from inference.graph.state import GraphState
from inference.graph.prompt_templates import format_template
from inference.llm import call_llm
from retrieval.confidence import get_confidence_for_chunks
from retrieval.db_utils import get_document_title

load_dotenv()
logger = logging.getLogger(__name__)
agent_log = get_agent_logger()


MAX_CONTEXT_CHUNKS = int(os.getenv('MAX_CONTEXT_CHUNKS', '24'))  # Increased to allow more context for verbose documents
MAX_CHUNKS_PER_DOC = int(os.getenv('MAX_CHUNKS_PER_DOC', '6'))  # Increased from 2 to allow more chunks per document for long/verbose docs

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
        logger.warning("=" * 60)
        logger.warning("SYNTHESIZER: No context chunks available - abstaining")
        logger.warning(f"Total evidence chunks: {len(evidence)}")
        logger.warning(f"Selected doc IDs: {selected_doc_ids}")
        logger.warning(f"Evidence chunks detail:")
        for idx, chunk in enumerate(evidence):
            logger.warning(f"  Chunk {idx}: doc_id={chunk.get('doc_id')}, chunk_id={chunk.get('chunk_id')}, p0={chunk.get('p0')}, p1={chunk.get('p1')}")
        logger.warning("=" * 60)
        agent_log.log_step(
            node="synthesizer",
            action="abstain_no_context",
            question=question_text,
            num_chunks=len(evidence),
            confidence=0.0,
            iterations=state.get("iterations", 0),
            metadata={
                "reason": "No context chunks after selection",
                "total_evidence": len(evidence),
                "selected_doc_ids": selected_doc_ids,
            }
        )
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
    doc_order_instruction = "Documents to use for your response:\n" + "\n".join(doc_order_lines) if doc_order_lines else ""

    question_for_confidence = state.get("question", "") or ""
    
    # ENHANCEMENT: When document(s) are explicitly selected or attached, use lower confidence threshold
    # This handles ambiguous queries like "share the details of this document"
    # Check for explicit selection via:
    # - selected_doc_ids: User explicitly selected document(s) in UI
    # - uploaded_doc_ids: User attached/uploaded document(s) 
    # - doc_id: Document from ingestion/previous query
    # NOTE: This does NOT apply to cross-doc search (when cross_doc=True and no specific docs selected)
    #       Cross-doc search uses default threshold since user hasn't explicitly selected documents
    cross_doc = state.get('cross_doc', False)
    is_explicit_doc_selection = (
        (selected_doc_ids and len(selected_doc_ids) > 0) or
        (uploaded_doc_ids and len(uploaded_doc_ids) > 0) or
        (doc_id is not None)
    )
    # Only use explicit selection threshold if NOT doing cross-doc search without specific docs
    # If cross_doc=True and no specific docs selected, use default threshold
    if cross_doc and not is_explicit_doc_selection:
        is_explicit_doc_selection = False  # Cross-doc without specific selection = default threshold
    
    # Get confidence thresholds from environment variables with sensible defaults
    # Default threshold: 40% for general queries
    # Explicit selection threshold: Uses THRESH (0.30) converted to percentage (30.0%)
    #   THRESH is the critic's chunk strength threshold, which aligns with explicit doc selection confidence
    from inference.graph.constants import THRESH
    default_threshold = float(os.getenv("SYNTHESIZER_CONFIDENCE_THRESHOLD_DEFAULT", "40.0"))
    # Convert THRESH (0.30) to percentage (30.0%) for explicit selection threshold
    # Handle case where env var might contain {THRESH} placeholder or use default
    explicit_selection_env = os.getenv("SYNTHESIZER_CONFIDENCE_THRESHOLD_EXPLICIT_SELECTION")
    if explicit_selection_env and explicit_selection_env.strip() not in ["{THRESH}", "{THRESH}*100"]:
        try:
            explicit_selection_threshold = float(explicit_selection_env)
        except ValueError:
            # If conversion fails, fall back to THRESH * 100
            explicit_selection_threshold = THRESH * 100
    else:
        # Default to THRESH * 100 (30.0%)
        explicit_selection_threshold = THRESH * 100
    
    # Adjust confidence threshold for explicit document selection
    # When user explicitly selects/attaches document(s), they want analysis even if query is ambiguous
    # IMPORTANT: This threshold only affects PRE-LLM abstention. The LLM can still return "I don't know"
    #            which will be detected by citation_pruner and handled appropriately.
    confidence_threshold = explicit_selection_threshold if is_explicit_doc_selection else default_threshold
    
    logger.info(f"Confidence threshold: {confidence_threshold:.1f}% (explicit_selection={is_explicit_doc_selection}, "
                f"cross_doc={cross_doc}, "
                f"selected_docs={len(selected_doc_ids) if selected_doc_ids else 0}, "
                f"uploaded_docs={len(uploaded_doc_ids) if uploaded_doc_ids else 0}, "
                f"doc_id={'present' if doc_id else 'none'})")
    
    conf_result = get_confidence_for_chunks(ctx_evs, query=question_for_confidence)
    overall_confidence = conf_result["confidence"]
    overall_probability = conf_result["probability"]
    action = conf_result["action"]
    logger.info(
        "Confidence %.2f%% (probability=%.3f) action=%s (threshold=%.1f%%, explicit_selection=%s)",
        overall_confidence,
        overall_probability,
        action,
        confidence_threshold,
        is_explicit_doc_selection,
    )

    # Pre-LLM confidence check: If confidence is too low, abstain BEFORE calling LLM
    # This prevents wasting tokens on queries that are unlikely to succeed
    # NOTE: Even if we pass this check and call the LLM, the LLM can still return "I don't know"
    #       which will be detected by citation_pruner and handled appropriately (clears citations, etc.)
    if action == "abstain" or overall_confidence < confidence_threshold:
        logger.warning("=" * 60)
        logger.warning("SYNTHESIZER ABSTAINING - CONFIDENCE TOO LOW")
        logger.warning("=" * 60)
        logger.warning(f"Question: {question_text}")
        logger.warning(f"Action from confidence check: {action}")
        logger.warning(f"Overall confidence: {overall_confidence:.2f}%")
        logger.warning(f"Confidence threshold: {confidence_threshold:.1f}%")
        logger.warning(f"Context chunks available: {len(ctx_evs)}")
        logger.warning(f"Top doc IDs: {top_doc_ids}")
        logger.warning(f"Selected doc IDs: {selected_doc_ids}")
        logger.warning("Reason: Confidence below threshold or action='abstain'")
        logger.warning("=" * 60)
        logger.info("Returning abstain result")
        agent_log.log_step(
            node="synthesizer",
            action="abstain_low_confidence",
            question=question_text,
            num_chunks=len(ctx_evs),
            confidence=overall_confidence,
            iterations=state.get("iterations", 0),
            metadata={
                "reason": "Confidence below threshold or action='abstain'",
                "confidence": overall_confidence,
                "probability": overall_probability,
                "action": action,
                "threshold": confidence_threshold,
                "context_chunks": len(ctx_evs),
                "top_doc_ids": top_doc_ids,
                "selected_doc_ids": selected_doc_ids,
            }
        )
        abstain_result: Dict[str, Any] = {
            "answer": "I don't know.",
            "confidence": overall_confidence,
            "action": "abstain",
            "doc_ids": [],
            "pages": [],
        }
        return cast(GraphState, abstain_result)

    # Assign alphabetic citations [A], [B], [C] to chunks in order they appear in ctx_evs
    # This preserves retrieval order and allows tracking which chunks are used
    chunk_to_letter: Dict[str, str] = {}  # chunk_id -> letter
    letter_to_doc_prefix: Dict[str, str] = {}  # letter -> doc_prefix
    letter_to_chunk: Dict[str, EvidenceChunk] = {}  # letter -> chunk (for confidence tracking)
    
    import string
    letters = string.ascii_uppercase
    
    for idx, chunk in enumerate(ctx_evs):
        chunk_id = chunk.get("chunk_id", "")
        doc_id = chunk.get("doc_id", "")
        doc_prefix = doc_id[:8] if doc_id else ""
        
        if chunk_id and idx < len(letters):
            letter = letters[idx]
            chunk_to_letter[chunk_id] = letter
            letter_to_chunk[letter] = chunk
            if doc_prefix:
                letter_to_doc_prefix[letter] = doc_prefix
    
    # Build document reference list for LLM prompt with alphabetic citations
    doc_reference_list = ""
    if ctx_evs:
        doc_reference_list = "\n\nAvailable Chunks (use alphabetic citations when referencing):\n"
        for idx, chunk in enumerate(ctx_evs[:26]):  # Limit to 26 chunks (A-Z)
            chunk_id = chunk.get("chunk_id", "")
            doc_id = chunk.get("doc_id", "")
            doc_prefix = doc_id[:8] if doc_id else "unknown"
            doc_title = get_document_title(doc_id) if doc_id else "Unknown"
            letter = letters[idx] if idx < len(letters) else "?"
            
            # Get chunk preview
            chunk_text = str(chunk.get("text", ""))[:100].replace("\n", " ")
            doc_reference_list += f"[{letter}] {doc_title} ({doc_prefix}): {chunk_text}...\n"
        
        doc_reference_list += "\nWhen you reference information from a chunk in your answer, use the alphabetic citation [A], [B], [C], etc. corresponding to the chunk letter above.\n"
        doc_reference_list += "Example: If discussing content from chunk [A], cite it as [A] at the end of the relevant sentence or paragraph."

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
    
    # Build Sources format example showing alphabetic citations with DOC prefixes
    # Format: [A] [DOC: prefix], [B] [DOC: prefix] in order of first use
    sources_example_lines = []
    for idx, chunk in enumerate(ctx_evs[:5]):  # Show first 5 as example
        if idx < len(letters):
            letter = letters[idx]
            doc_id = chunk.get("doc_id", "")
            doc_prefix = doc_id[:8] if doc_id else "unknown"
            sources_example_lines.append(f"- [{letter}] [DOC: {doc_prefix}]")
    
    sources_example = "\n".join(sources_example_lines) if sources_example_lines else "- [A] [DOC: a1b2c3d4]"
    format = f"""\n\nSources:\n{sources_example}\n\nList sources using alphabetic citations [A], [B], [C], etc. in the order you first mentioned them in your answer. Each letter corresponds to a chunk, followed by [DOC: prefix] where prefix is the 8-character document ID prefix."""
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

    if is_content_request and is_multi_doc_query:
        prompt = format_template(
            "synthesizer_content_multi_doc",
            doc_reference_list=doc_reference_list,
            num_documents=len(selected_doc_ids),
            citation_format=format,
            order_block=order_block,
            question_lower=question_lower,
            context=context
        )
    else:
        prompt = format_template(
            "synthesizer_standard",
            doc_reference_list=doc_reference_list,
            question_text=question_text,
            citation_format=format,
            order_block=order_block,
            question_lower=question_lower,
            num_documents=len(selected_doc_ids),
            context=context
        )

    # Build ranked citations with confidence scores for chunks actually used in inference
    def calculate_chunk_confidence(chunk: EvidenceChunk) -> float:
        """Calculate per-chunk confidence score based on lex, vec, and ce scores."""
        lex_score = float(chunk.get("lex", 0.0) or 0.0)
        vec_score = float(chunk.get("vec", 0.0) or 0.0)
        ce_score = float(chunk.get("ce", 0.0) or 0.0)
        
        # Weighted combination: prioritize cross-encoder if available, otherwise use vec+lex
        if ce_score > 0:
            # Cross-encoder is most reliable, weight it higher
            chunk_confidence = (0.2 * lex_score + 0.3 * vec_score + 0.5 * ce_score) * 100
        else:
            # Fallback to vector + lexical combination
            chunk_confidence = (0.4 * lex_score + 0.6 * vec_score) * 100
        
        return chunk_confidence
    
    # Calculate confidence for each chunk and group by page for "Documents used for analysis"
    # Group chunks by (doc_id, page) to show confidence per page
    page_confidence_map: Dict[Tuple[str, Optional[int]], List[float]] = defaultdict(list)
    
    for chunk in ctx_evs:
        doc_id = chunk.get("doc_id", "")
        p0 = chunk.get("p0")
        page_key = (doc_id, p0)
        confidence = calculate_chunk_confidence(chunk)
        page_confidence_map[page_key].append(confidence)
    
    # Calculate average confidence per document (across all pages) for ranking
    # Group all confidences by document to calculate overall document contribution strength
    doc_all_confidences: Dict[str, List[float]] = defaultdict(list)
    for (doc_id, _), confidences in page_confidence_map.items():
        doc_all_confidences[doc_id].extend(confidences)
    
    # Calculate average confidence per document (contribution strength)
    doc_avg_confidence: Dict[str, float] = {}
    for doc_id, all_confidences in doc_all_confidences.items():
        if all_confidences:
            doc_avg_confidence[doc_id] = sum(all_confidences) / len(all_confidences)
        else:
            doc_avg_confidence[doc_id] = 0.0
    
    # Sort documents by their average contribution strength (confidence) - descending
    # This ensures the most relevant documents appear first
    sorted_docs_by_confidence = sorted(
        doc_avg_confidence.items(),
        key=lambda item: -item[1],  # Sort by confidence descending
    )
    
    # Create a mapping from doc_id to rank based on contribution strength
    doc_rank_map: Dict[str, int] = {}
    for rank, (doc_id, _) in enumerate(sorted_docs_by_confidence, start=1):
        doc_rank_map[doc_id] = rank
    
    # Calculate average confidence per page and build page-level citations
    page_citations = []
    for (doc_id, page_num), confidences in page_confidence_map.items():
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Get document title
        doc_title = None
        if doc_id:
            doc_title = get_document_title(doc_id)
        
        # Format page
        if isinstance(page_num, int):
            page_str = f"p{page_num}"
        else:
            page_str = "p?"
        
        # Get doc rank based on contribution strength (not top_doc_ids order)
        doc_rank = doc_rank_map.get(doc_id)
        
        if doc_rank:
            doc_label = doc_title if doc_title else doc_id[:8]
            # Store: (doc_rank, doc_label, page_str, avg_confidence, doc_id, page_num)
            # page_num is kept for secondary sorting by page number when confidence is equal
            page_citations.append((doc_rank, doc_label, page_str, avg_confidence, doc_id, page_num))
    
    # Sort by document rank (contribution strength), then by contribution strength descending within each document
    # If contribution strength is equal, sort by page number ascending
    page_citations.sort(key=lambda x: (
        x[0],  # Document rank
        -x[3],  # Contribution strength (negative for descending)
        x[5] if isinstance(x[5], int) else 999  # Page number (ascending) as tiebreaker
    ))
    
    # Build ranked citations for "Documents used for analysis" section
    ranked_citations = []
    for doc_rank, doc_label, page_str, avg_confidence, doc_id, _ in page_citations:
        citation = f"[{doc_rank}] \"{doc_label}\" - Page: {page_str} - (contribution strength: {avg_confidence:.1f}%)"
        ranked_citations.append(citation)
        logger.debug(f"Page citation: {citation}")
    
    logger.info(f"Built {len(ranked_citations)} page-level citations from {len(ctx_evs)} context chunks")
    logger.info("Invoking LLM for synthesis")
    logger.info(f"Prompt length: {len(prompt)} characters")
    logger.info(f"Context chunks: {len(ctx_evs)}, Top doc IDs: {top_doc_ids}")
    
    # Estimate input tokens (rough approximation: ~4 characters per token)
    system_prompt = "You write precise, grounded answers. Avoid speculation and keep sources aligned. Answer with I dont know if you cannot ground your answer."
    estimated_input_tokens = (len(system_prompt) + len(prompt)) // 4
    logger.info(f"Estimated input tokens: ~{estimated_input_tokens} (based on character count)")
    
    llm_response, token_info = call_llm(
        system_prompt,
        [{"role": "user", "content": prompt}],
        max_tokens=1800,
        temperature=0.2,
    )
    
    # Log token usage
    input_tokens = token_info.get("input_tokens", 0)
    output_tokens = token_info.get("output_tokens", 0)
    total_tokens = token_info.get("total_tokens", 0)
    logger.info("=" * 60)
    logger.info("TOKEN USAGE:")
    logger.info(f"  Input tokens: {input_tokens}")
    logger.info(f"  Output tokens: {output_tokens}")
    logger.info(f"  Total tokens: {total_tokens}")
    if estimated_input_tokens > 0 and input_tokens > 0:
        logger.info(f"  Estimation accuracy: {abs(estimated_input_tokens - input_tokens) / input_tokens * 100:.1f}% difference")
    logger.info("=" * 60)
    
    # Log raw LLM response for debugging
    raw_answer = llm_response.strip()
    logger.info("=" * 60)
    logger.info("RAW LLM RESPONSE FROM SYNTHESIZER:")
    logger.info(f"Length: {len(raw_answer)} characters")
    logger.info(f"First 500 chars: {raw_answer[:500]}")
    logger.info(f"Full response: {raw_answer}")
    logger.info("=" * 60)
    
    answer_text = raw_answer
    
    # Add calculated and ranked citations showing which pages were most relevant for the answer
    if ranked_citations:
        answer_text += "\n\nDocuments used for analysis (ranked by contribution strength):\n" + "\n".join(ranked_citations)
    
    
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
    # Synthesizer only determines if it can answer or must abstain
    # Clarification/refinement is handled by compressor/critic loop
    final_action = "abstain" if action == "abstain" else "answer"
    
    # Pass through all top_doc_ids - citation_pruner will prune based on LLM answer
    # Also pass letter mappings for tracking chunk usage and confidence
    result_payload: Dict[str, Any] = {
        "answer": final_answer,
        "confidence": overall_confidence,
        "action": final_action,
        "doc_ids": top_doc_ids,  # Pass all documents - pruner will filter
        "pages": sorted(set(page_numbers)),
        "chunk_to_letter": chunk_to_letter,  # For tracking which chunks were used
        "letter_to_doc_prefix": letter_to_doc_prefix,  # For mapping letters to doc prefixes
        "letter_to_chunk": {k: v.get("chunk_id", "") for k, v in letter_to_chunk.items()},  # For confidence tracking
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