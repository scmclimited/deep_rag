"""
Citation Pruner node: Post-synthesizer node that prunes citations and replaces document IDs with titles.
This node processes the LLM response to extract document references and ensures only referenced documents
are included in the sources list.
"""
import logging
import re
from typing import Any, Dict, List, Match, Optional, Set, cast
from inference.graph.state import GraphState
from inference.graph.agent_logger import get_agent_logger
from retrieval.db_utils import get_document_title

logger = logging.getLogger(__name__)
agent_log = get_agent_logger()


def _normalize_for_match(text: str) -> str:
    """Normalize text for matching (lowercase, strip whitespace)."""
    return text.lower().strip() if text else ""


def _check_idont_know(answer: str) -> bool:
    """Check if the answer is "I don't know" or similar negative response."""
    normalized = _normalize_for_match(answer)
    
    # Exact matches
    if normalized == "i don't know" or normalized == "i dont know" or normalized == "i do not know":
        return True
    
    # Check for negative response patterns
    negative_patterns = [
        r"^i\s+don'?t\s+know",
        r"^i\s+do\s+not\s+know",
        r"does\s+not\s+contain\s+the\s+answer",
        r"does\s+not\s+contain\s+the\s+information",
        r"does\s+not\s+provide\s+the\s+answer",
        r"no\s+answer\s+is\s+available",
        r"no\s+relevant\s+information",
        r"cannot\s+determine\s+from\s+the\s+document",
        r"cannot\s+find\s+this\s+information",
        r"not\s+provided\s+in\s+the\s+document",
        r"document\s+does\s+not\s+provide",
        r"document\s+does\s+not\s+mention",
        r"not\s+enough\s+information\s+in\s+the\s+document",
        r"context\s+does\s+not\s+contain",
        r"no\s+supportive\s+evidence\s+in\s+the\s+context",
    ]
    
    for pattern in negative_patterns:
        if re.search(pattern, normalized, re.IGNORECASE):
            return True
    
    return False


def _extract_doc_references(answer: str) -> Set[str]:
    """
    Extract document references from answer text.
    Looks for patterns like:
    - [DOC {doc[:8]}] or [DOC doc[:8]]
    - DOC {doc[:8]} or DOC doc[:8]
    - Document {doc[:8]} or Document doc[:8] (case-insensitive)
    - doc:doc[:8] or doc: {doc[:8]}
    
    Returns:
        Set of 8-character document ID prefixes found in the answer
    """
    doc_refs: Set[str] = set()
    
    # Pattern 1: [DOC {doc[:8]}] or [DOC doc[:8]]
    pattern1 = r'\[DOC\s+\{?([a-f0-9]{8})\}?\]'
    matches1 = re.findall(pattern1, answer, re.IGNORECASE)
    doc_refs.update(matches1)
    
    # Pattern 2: DOC {doc[:8]} or DOC doc[:8] (without brackets)
    pattern2 = r'DOC\s+\{?([a-f0-9]{8})\}?'
    matches2 = re.findall(pattern2, answer, re.IGNORECASE)
    doc_refs.update(matches2)
    
    # Pattern 3: Document {doc[:8]} or Document doc[:8] (case-insensitive)
    # Make sure we match "Document" as a word (not part of another word)
    pattern3 = r'\bDocument\s+\{?([a-f0-9]{8})\}?'
    matches3 = re.findall(pattern3, answer, re.IGNORECASE)
    doc_refs.update(matches3)
    logger.debug(f"Pattern 3 (Document) matches: {matches3}")
    
    # Pattern 4: doc:doc[:8] or doc: {doc[:8]}
    pattern4 = r'doc:\s*\{?([a-f0-9]{8})\}?'
    matches4 = re.findall(pattern4, answer, re.IGNORECASE)
    doc_refs.update(matches4)
    
    logger.debug(f"Extracted document references from answer: {doc_refs}")
    return doc_refs


def _match_doc_ids_by_prefix(doc_refs: Set[str], available_doc_ids: List[str]) -> Set[str]:
    """
    Match 8-character prefixes to full document IDs.
    
    Args:
        doc_refs: Set of 8-character document ID prefixes
        available_doc_ids: List of full document IDs available in context
        
    Returns:
        Set of full document IDs that match the prefixes
    """
    matched_doc_ids: Set[str] = set()
    doc_refs_lower = {ref.lower() for ref in doc_refs}
    
    logger.debug(f"Matching {len(doc_refs)} reference(s) against {len(available_doc_ids)} available doc_id(s)")
    
    for doc_id in available_doc_ids:
        doc_id_prefix = doc_id[:8].lower()
        if doc_id_prefix in doc_refs_lower:
            matched_doc_ids.add(doc_id)
            logger.info(f"✓ Matched prefix '{doc_id_prefix}' to full doc_id: {doc_id}")
        else:
            logger.debug(f"✗ Prefix '{doc_id_prefix}' not in references {doc_refs_lower}")
    
    if not matched_doc_ids and doc_refs:
        logger.warning(f"No matches found! References: {doc_refs_lower}, Available prefixes: {[d[:8].lower() for d in available_doc_ids]}")
    
    return matched_doc_ids


def _replace_doc_citations(answer: str, doc_id_to_title: Dict[str, str]) -> str:
    """
    Replace document citations in the answer with document titles.
    
    Patterns replaced:
    - [DOC {doc[:8]}] -> [Document Title]
    - [DOC doc[:8]] -> [Document Title]
    - DOC {doc[:8]} -> Document Title
    - DOC doc[:8] -> Document Title
    - Document {doc[:8]} -> Document Title
    - Document doc[:8] -> Document Title
    - doc:doc[:8] -> Document Title
    - doc: {doc[:8]} -> Document Title
    
    Args:
        answer: Original answer text
        doc_id_to_title: Dictionary mapping full doc_id to document title
        
    Returns:
        Answer text with citations replaced by titles
    """
    result = answer
    
    # Build a map of 8-character prefixes to titles
    prefix_to_title: Dict[str, str] = {}
    for doc_id, title in doc_id_to_title.items():
        prefix = doc_id[:8].lower()
        if title:
            prefix_to_title[prefix] = title
    
    # Replace patterns with titles
    # Pattern 1: [DOC {doc[:8]}] or [DOC doc[:8]]
    def replace_bracketed(match: Match[str]) -> str:
        prefix = match.group(1).lower()
        title = prefix_to_title.get(prefix)
        if title:
            return f"[{title}]"
        return match.group(0)  # Keep original if title not found
    
    result = re.sub(r'\[DOC\s+\{?([a-f0-9]{8})\}?\]', replace_bracketed, result, flags=re.IGNORECASE)
    
    # Pattern 2: DOC {doc[:8]} or DOC doc[:8] (without brackets)
    def replace_unbracketed(match: Match[str]) -> str:
        prefix = match.group(1).lower()
        title = prefix_to_title.get(prefix)
        if title:
            return title
        return match.group(0)  # Keep original if title not found
    
    result = re.sub(r'DOC\s+\{?([a-f0-9]{8})\}?', replace_unbracketed, result, flags=re.IGNORECASE)
    
    # Pattern 3: Document {doc[:8]} or Document doc[:8] (case-insensitive)
    # Match "Document" as a word boundary to avoid matching "Documentation" etc.
    def replace_document_word(match: Match[str]) -> str:
        prefix = match.group(1).lower()
        title = prefix_to_title.get(prefix)
        if title:
            return title
        return match.group(0)  # Keep original if title not found
    
    result = re.sub(r'\bDocument\s+\{?([a-f0-9]{8})\}?', replace_document_word, result, flags=re.IGNORECASE)
    
    # Pattern 4: doc:doc[:8] or doc: {doc[:8]}
    def replace_doc_colon(match: Match[str]) -> str:
        prefix = match.group(1).lower()
        title = prefix_to_title.get(prefix)
        if title:
            return title
        return match.group(0)  # Keep original if title not found
    
    result = re.sub(r'doc:\s*\{?([a-f0-9]{8})\}?', replace_doc_colon, result, flags=re.IGNORECASE)
    
    return result


def _build_document_map(doc_ids: List[str]) -> Dict[str, Optional[str]]:
    """
    Build a map of doc_id to document title.
    
    Args:
        doc_ids: List of document IDs
        
    Returns:
        Dictionary mapping doc_id to document title (or None if not found)
    """
    doc_map: Dict[str, Optional[str]] = {}
    for doc_id in doc_ids:
        title = get_document_title(doc_id)
        doc_map[doc_id] = title
        logger.debug(f"Mapped doc_id {doc_id[:8]}... to title: {title}")
    return doc_map


def _prune_citations(citations: List[str], used_doc_ids: Set[str], doc_id_to_title: Dict[str, Optional[str]]) -> List[str]:
    """
    Prune citations to only include documents that were actually used.
    Also replace doc IDs with titles in citation strings.
    
    Args:
        citations: Original citations list
        used_doc_ids: Set of document IDs that were actually referenced
        doc_id_to_title: Dictionary mapping doc_id to title
        
    Returns:
        Pruned and updated citations list
    """
    pruned_citations: List[str] = []
    
    for citation in citations:
        # Extract doc_id from citation (format: "[{idx}] doc:{doc_id} {page_str} (confidence: {conf}%)")
        doc_match = re.search(r'doc:([a-f0-9-]+)', citation, re.IGNORECASE)
        if doc_match:
            doc_id = doc_match.group(1)
            if doc_id in used_doc_ids:
                # Replace doc_id with title in citation
                title = doc_id_to_title.get(doc_id)
                if title:
                    # Replace "doc:{doc_id}" with title
                    updated_citation = re.sub(
                        r'doc:[a-f0-9-]+',
                        title,
                        citation,
                        flags=re.IGNORECASE
                    )
                    pruned_citations.append(updated_citation)
                else:
                    # Keep original if title not found
                    pruned_citations.append(citation)
    
    return pruned_citations


def node_citation_pruner(state: GraphState) -> GraphState:
    """
    Citation Pruner node: Post-synthesizer processing.
    
    Responsibilities:
    1. Check for "I don't know" responses and clear all sources if found
    2. Extract document references from LLM answer (format: [DOC {doc[:8]}])
    3. Replace document ID citations with document titles
    4. Prune sources list to only include documents referenced in the answer
    5. Update citations to use document titles instead of IDs
    
    Args:
        state: GraphState from synthesizer
        
    Returns:
        Updated GraphState with pruned citations and replaced document references
    """
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Citation Pruner - Processing citations and pruning sources")
    logger.info("-" * 40)
    
    answer = state.get("answer", "")
    doc_ids = state.get("doc_ids", [])
    citations = state.get("citations", [])
    confidence = state.get("confidence", 0.0)
    action = state.get("action", "answer")
    
    logger.info(f"Input state: answer_length={len(answer)}, doc_ids={len(doc_ids)}, citations={len(citations)}")
    
    # Step 1: Check for "I don't know" response
    if _check_idont_know(answer):
        logger.info("Detected 'I don't know' response - clearing all sources and citations")
        pruned_result: Dict[str, Any] = {
            "answer": "I don't know.",
            "confidence": min(confidence, 40.0),
            "action": "abstain",
            "doc_ids": [],
            "pages": [],
            "citations": [],
        }
        
        agent_log.log_step(
            node="citation_pruner",
            action="prune_abstain",
            question=state.get("question", ""),
            answer="I don't know.",
            num_chunks=0,
            pages=None,
            confidence=min(confidence, 40.0),
            iterations=state.get("iterations", 0),
            metadata={
                "reason": "I don't know detected",
                "original_doc_ids": doc_ids,
            },
        )
        
        return cast(GraphState, pruned_result)
    
    # Step 2: Extract document references from answer
    doc_refs = _extract_doc_references(answer)
    logger.info(f"Extracted {len(doc_refs)} document reference(s) from answer: {[ref for ref in doc_refs]}")
    if doc_refs:
        logger.info(f"Document reference prefixes found: {list(doc_refs)}")
    else:
        logger.warning(f"No document references found in answer. Answer preview: {answer[:500]}")
        logger.debug(f"Available doc_ids to match against: {[d[:8] for d in doc_ids]}")
    
    # Step 3: Match references to full document IDs
    used_doc_ids: Set[str] = _match_doc_ids_by_prefix(doc_refs, doc_ids) if doc_refs else set()
    
    # CRITICAL: Only keep documents that were explicitly referenced in the answer
    # No fallback - if LLM didn't cite it, don't include it
    if not used_doc_ids:
        logger.warning("No explicit document references found in answer - clearing all sources")
        used_doc_ids = set()
    else:
        logger.info(f"Found explicit document references: {[d[:8] + '...' for d in used_doc_ids]}")
    
    logger.info(f"Matched {len(used_doc_ids)} document(s) to references: {[d[:8] + '...' for d in used_doc_ids]}")
    
    # Step 4: Build document title map for ALL available docs (for replacement)
    # But we'll only return the used ones
    all_doc_id_to_title = _build_document_map(doc_ids)
    
    # Step 5: Replace document citations in answer with titles
    # Use all docs for replacement (in case answer mentions docs not in used set)
    updated_answer = _replace_doc_citations(answer, {k: v for k, v in all_doc_id_to_title.items() if v})
    logger.info(f"Replaced document citations in answer (length: {len(updated_answer)})")
    
    # Build title map only for used documents
    doc_id_to_title = {k: v for k, v in all_doc_id_to_title.items() if k in used_doc_ids}
    
    # Step 6: Prune citations to only include used documents
    pruned_citations = _prune_citations(citations, used_doc_ids, doc_id_to_title)
    logger.info(f"Pruned citations from {len(citations)} to {len(pruned_citations)}")
    
    # Step 7: Update pages to only include pages from used documents
    # (Pages are already filtered in synthesizer, but we ensure consistency)
    pages = state.get("pages", [])
    
    # Step 8: Update the answer's Sources section if it exists
    # Remove old Sources section and add new one with pruned citations
    if "Sources:" in updated_answer:
        # Remove existing Sources section
        sources_pattern = r'\n\nSources:.*$'
        updated_answer = re.sub(sources_pattern, '', updated_answer, flags=re.DOTALL)
    
    # Add updated Sources section if we have citations
    if pruned_citations:
        updated_answer = updated_answer.rstrip()
        updated_answer += "\n\nSources: " + ", ".join(pruned_citations)
    
    # Step 9: Build document map with "used" status for frontend
    doc_map: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        is_used = doc_id in used_doc_ids
        title = all_doc_id_to_title.get(doc_id)
        doc_map.append({
            "doc_id": doc_id,
            "title": title,
            "used": is_used
        })
    
    # Step 10: Build result payload
    result_payload: Dict[str, Any] = {
        "answer": updated_answer,
        "confidence": confidence,
        "action": action,
        "doc_ids": list(used_doc_ids),  # Only include documents that were referenced
        "pages": pages,
        "doc_map": doc_map,  # Document map with "used" status for frontend
    }
    
    if pruned_citations:
        result_payload["citations"] = pruned_citations
    
    # Preserve doc_id if it's in the used set
    primary_doc_id = state.get("doc_id")
    if primary_doc_id and primary_doc_id in used_doc_ids:
        result_payload["doc_id"] = primary_doc_id
    
    logger.info(f"Citation pruning complete: {len(used_doc_ids)} document(s) retained")
    logger.info(f"Updated answer preview: {updated_answer[:200]}...")
    logger.info("-" * 40)
    
    agent_log.log_step(
        node="citation_pruner",
        action="prune_citations",
        question=state.get("question", ""),
        answer=updated_answer,
        num_chunks=len(state.get("evidence", [])),
        pages=pages if pages else None,
        confidence=confidence,
        iterations=state.get("iterations", 0),
        metadata={
            "original_doc_ids": doc_ids,
            "pruned_doc_ids": list(used_doc_ids),
            "doc_refs_found": list(doc_refs),
            "citations_before": len(citations),
            "citations_after": len(pruned_citations),
            "doc_titles": {k: v for k, v in doc_id_to_title.items() if v},
        },
    )
    
    return cast(GraphState, result_payload)

