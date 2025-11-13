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
    question = state.get("question", "")
    
    logger.info(f"Input state: answer_length={len(answer)}, doc_ids={len(doc_ids)}, citations={len(citations)}")
    logger.info(f"Question: {question}")
    logger.info(f"Confidence: {confidence:.2f}%, Action: {action}")
    logger.info(f"Answer preview (first 500 chars): {answer[:500]}")
    
    # Step 1: Check for "I don't know" response
    is_idont_know = _check_idont_know(answer)
    if is_idont_know:
        logger.warning("=" * 60)
        logger.warning("DETECTED 'I DON'T KNOW' RESPONSE IN CITATION_PRUNER")
        logger.warning("=" * 60)
        logger.warning(f"Question: {question}")
        logger.warning(f"Answer (full): {answer}")
        logger.warning(f"Answer length: {len(answer)} characters")
        logger.warning(f"Confidence from synthesizer: {confidence:.2f}%")
        logger.warning(f"Action from synthesizer: {action}")
        logger.warning(f"Document IDs provided: {doc_ids}")
        logger.warning(f"Number of citations: {len(citations)}")
        logger.warning("This may indicate:")
        logger.warning("  1. LLM generated 'I don't know' despite having context")
        logger.warning("  2. Confidence threshold was too low (< 40%)")
        logger.warning("  3. Context was insufficient or irrelevant")
        logger.warning("  4. Prompt may have been too restrictive")
        logger.warning("=" * 60)
        logger.info("Clearing all sources and citations due to 'I don't know' detection")
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
    
    # Step 2: Extract document references from answer body
    doc_refs = _extract_doc_references(answer)
    logger.info(f"Extracted {len(doc_refs)} document reference(s) from answer body: {[ref for ref in doc_refs]}")
    
    # Step 2a: Also extract from alphabetic citations in answer body using letter_to_doc_prefix
    letter_to_doc_prefix = state.get("letter_to_doc_prefix", {})
    if letter_to_doc_prefix:
        logger.info(f"Found letter_to_doc_prefix mapping: {letter_to_doc_prefix}")
        # Extract alphabetic citations like [B], [G], [M] from answer body
        alphabetic_citations = re.findall(r'\[([A-Z])\]', answer)
        if alphabetic_citations:
            logger.info(f"Found alphabetic citations in answer body: {set(alphabetic_citations)}")
            # Map letters to doc prefixes
            for letter in set(alphabetic_citations):
                if letter in letter_to_doc_prefix:
                    doc_prefix = letter_to_doc_prefix[letter].lower()
                    doc_refs.add(doc_prefix)
                    logger.debug(f"Mapped citation [{letter}] to doc prefix: {doc_prefix}")
    
    # Step 2b: Also extract document references from Sources section if present
    if "Sources:" in answer:
        # Match Sources section with flexible newline handling (1 or 2 newlines)
        sources_match = re.search(r'\n+Sources:.*$', answer, re.DOTALL)
        if sources_match:
            sources_text = sources_match.group(0)
            # Extract [DOC: 16a68247] patterns from Sources section
            sources_doc_refs = re.findall(r'\[DOC:\s*([a-f0-9]{8})\]\s*', sources_text, re.IGNORECASE)
            if sources_doc_refs:
                doc_refs.update([ref.lower() for ref in sources_doc_refs])
                logger.info(f"Extracted {len(sources_doc_refs)} document reference(s) from Sources section: {sources_doc_refs}")
    
    if doc_refs:
        logger.info(f"Total document reference prefixes found: {list(doc_refs)}")
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
    
    # Step 8: Extract and preserve the LLM's Sources section AND "Documents used for analysis" section
    # The LLM generates Sources in format: "- [B] [DOC: 16a68247]" with alphabetic citations
    # We need to extract this section, filter to only used documents, and preserve the format
    # Also preserve the "Documents used for analysis (sorted bycontribution strength):" section with confidence scores
    sources_section = ""
    documents_analysis_section = ""
    
    if "Sources:" in answer:
        # Extract the Sources section from the original answer (before citation replacement)
        # Stop at "Documents used for analysis" if it exists, otherwise match to end
        # Try multiple patterns to handle different formats
        sources_text = None
        
        # Pattern 1: Sources: with preceding newlines, stop at "Documents used for analysis"
        sources_match = re.search(r'\n+Sources:.*?(?=\n+Documents used for analysis|$)', answer, re.DOTALL)
        if sources_match:
            sources_text = sources_match.group(0)
            logger.info(f"Found Sources section (pattern 1): {sources_text[:200]}...")
        else:
            # Pattern 2: Sources: at start of line (with MULTILINE flag)
            sources_match_alt = re.search(r'^Sources:.*?(?=\n+Documents used for analysis|$)', answer, re.MULTILINE | re.DOTALL)
            if sources_match_alt:
                sources_text = sources_match_alt.group(0)
                logger.info(f"Found Sources section (pattern 2): {sources_text[:200]}...")
            else:
                # Pattern 3: Just find "Sources:" and everything after until "Documents used for analysis" or end
                sources_idx = answer.find("Sources:")
                if sources_idx >= 0:
                    docs_idx = answer.find("Documents used for analysis", sources_idx)
                    if docs_idx >= 0:
                        sources_text = answer[sources_idx:docs_idx].rstrip()
                    else:
                        sources_text = answer[sources_idx:].rstrip()
                    logger.info(f"Found Sources section (pattern 3 - substring): {sources_text[:200]}...")
                else:
                    logger.warning(f"Sources: found in answer but all extraction patterns failed. Answer snippet: {answer[max(0, answer.find('Sources:')-50):answer.find('Sources:')+200]}")
        
        if sources_text:
            # letter_to_doc_prefix was already retrieved above
            logger.debug(f"letter_to_doc_prefix mapping: {letter_to_doc_prefix}")
            
            # Parse alphabetic citations from Sources section: "- [B] [DOC: 16a68247]"
            sources_lines = []
            for line in sources_text.split('\n'):
                line = line.strip()
                if not line or line == "Sources:":
                    if line == "Sources:":
                        sources_lines.append("Sources:")
                    continue
                
                # Match pattern: "- [B] [DOC: 16a68247]" or "- [B] [DOC:16a68247]"
                citation_match = re.match(r'^-\s*\[([A-Z])\]\s*\[DOC:\s*([a-f0-9]{8})\]\s*$', line, re.IGNORECASE)
                if citation_match:
                    letter = citation_match.group(1).upper()
                    doc_prefix = citation_match.group(2).lower()
                    
                    # Check if this document was actually used
                    # First, verify the letter maps to the correct doc_prefix
                    expected_prefix = letter_to_doc_prefix.get(letter, "").lower()
                    if expected_prefix == doc_prefix:
                        # Find the full doc_id that matches this prefix
                        matching_doc_id = None
                        for doc_id in doc_ids:
                            if doc_id[:8].lower() == doc_prefix:
                                matching_doc_id = doc_id
                                break
                        
                        # Include if:
                        # 1. The doc_id is in used_doc_ids (explicitly referenced in answer), OR
                        # 2. The doc_id exists in doc_ids (available in context) and letter_to_doc_prefix is valid
                        # This ensures Sources section is preserved even if used_doc_ids is empty due to alphabetic citations
                        if matching_doc_id:
                            if matching_doc_id in used_doc_ids:
                                # Explicitly referenced - include it
                                sources_lines.append(line)
                                logger.debug(f"Including citation: {line} (doc_id: {matching_doc_id[:8]}... in used_doc_ids)")
                            elif letter_to_doc_prefix and letter in letter_to_doc_prefix:
                                # Valid letter mapping - include it (alphabetic citation was used in answer)
                                sources_lines.append(line)
                                logger.debug(f"Including citation: {line} (doc_id: {matching_doc_id[:8]}... via letter mapping)")
                            else:
                                logger.debug(f"Excluding citation: {line} (document not in used_doc_ids and no valid letter mapping)")
                        else:
                            logger.debug(f"Excluding citation: {line} (doc_id not found for prefix {doc_prefix})")
                    else:
                        # If letter_to_doc_prefix is empty, still include if doc_prefix is in used_doc_ids
                        # This handles the case where LLM generated Sources but letter mapping is missing
                        if not letter_to_doc_prefix or expected_prefix == "":
                            matching_doc_id = None
                            for doc_id in doc_ids:
                                if doc_id[:8].lower() == doc_prefix:
                                    matching_doc_id = doc_id
                                    break
                            if matching_doc_id and matching_doc_id in used_doc_ids:
                                sources_lines.append(line)
                                logger.debug(f"Including citation: {line} (doc_id: {matching_doc_id[:8]}... in used_doc_ids, no letter mapping)")
                            else:
                                logger.debug(f"Excluding citation: {line} (letter {letter} doesn't match expected prefix {expected_prefix} and doc not in used_doc_ids)")
                        else:
                            logger.debug(f"Excluding citation: {line} (letter {letter} doesn't match expected prefix {expected_prefix})")
                else:
                    # Keep non-citation lines (like "Sources:" header)
                    if line:
                        sources_lines.append(line)
            
            # Rebuild Sources section if we have any citations
            if len(sources_lines) > 1:  # More than just "Sources:"
                sources_section = "\n" + "\n".join(sources_lines)
                logger.info(f"Rebuilt Sources section with {len(sources_lines) - 1} citation(s): {sources_section[:200]}...")
            else:
                logger.warning(f"Sources section found but no valid citations after filtering. sources_lines={sources_lines}, letter_to_doc_prefix={letter_to_doc_prefix}, used_doc_ids={[d[:8] for d in used_doc_ids]}")
                # If we found Sources but filtered everything out, preserve the original
                # We'll replace [DOC: prefix] with titles later regardless
                if sources_text:
                    logger.info("Preserving Sources section as-is (will replace [DOC: prefix] with titles)")
                    original_sources_lines = []
                    for line in sources_text.split('\n'):
                        line = line.strip()
                        if line and (line == "Sources:" or re.match(r'^-\s*\[([A-Z])\]\s*\[DOC:', line, re.IGNORECASE)):
                            original_sources_lines.append(line)
                    if len(original_sources_lines) > 1:
                        sources_section = "\n" + "\n".join(original_sources_lines)
                        logger.info(f"Preserved original Sources section: {sources_section[:200]}...")
    
    # Extract "Documents used for analysis" section separately
    # This section contains confidence scores per page, so we must preserve it exactly as-is
    if "Documents used for analysis" in answer:
        # Extract the entire "Documents used for analysis" section (preserve confidence scores)
        # Use a more precise pattern to ensure we get everything including confidence scores
        docs_analysis_match = re.search(r'\n+Documents used for analysis.*$', answer, re.DOTALL)
        if docs_analysis_match:
            documents_analysis_section = docs_analysis_match.group(0)
            # Verify contribution strength scores are present (check for both old "confidence" and new "contribution strength")
            has_contribution = '(contribution strength:' in documents_analysis_section.lower() or 'contribution strength:' in documents_analysis_section.lower()
            has_confidence = '(confidence:' in documents_analysis_section.lower() or 'confidence:' in documents_analysis_section.lower()
            has_scores = has_contribution or has_confidence
            logger.info(f"Found 'Documents used for analysis' section (length: {len(documents_analysis_section)}, has_scores: {has_scores}): {documents_analysis_section[:300]}...")
            if not has_scores:
                logger.warning("'Documents used for analysis' section extracted but no contribution strength scores detected!")
    
    # Remove old Sources section and "Documents used for analysis" section from updated_answer if they exist
    if "Sources:" in updated_answer:
        # Match Sources section with flexible newline handling (1 or 2 newlines)
        # Stop at "Documents used for analysis" if present
        sources_pattern = r'\n+Sources:.*?(?=\n+Documents used for analysis|$)'
        updated_answer = re.sub(sources_pattern, '', updated_answer, flags=re.DOTALL)
    
    if "Documents used for analysis" in updated_answer:
        # Remove "Documents used for analysis" section (we'll add it back after Sources)
        docs_analysis_pattern = r'\n+Documents used for analysis.*$'
        updated_answer = re.sub(docs_analysis_pattern, '', updated_answer, flags=re.DOTALL)
    
    # Add preserved Sources section if we have one
    # Replace [DOC: prefix] with document titles in Sources section
    if sources_section:
        # Replace [DOC: prefix] patterns with document titles
        # Build prefix to title map from all_doc_id_to_title
        prefix_to_title: Dict[str, str] = {}
        for doc_id, title in all_doc_id_to_title.items():
            if title:
                prefix = doc_id[:8].lower()
                prefix_to_title[prefix] = title
        
        # Replace [DOC: prefix] with titles in Sources section
        # Format: "- [B] [DOC: 16a68247]" -> "- [B] Document Title\n"
        # Process line by line to ensure each citation is on its own line
        sources_lines_final = []
        for line in sources_section.split('\n'):
            line = line.rstrip()  # Remove trailing whitespace but preserve leading
            if not line or line == "Sources:":
                if line == "Sources:":
                    sources_lines_final.append("Sources:")
                continue
            
            # Match pattern: "- [B] [DOC: 16a68247]" or "- [B] [DOC:16a68247]"
            citation_match = re.match(r'^(-\s*\[([A-Z])\]\s*)\[DOC:\s*([a-f0-9]{8})\]\s*$', line, re.IGNORECASE)
            if citation_match:
                prefix = citation_match.group(3).lower()
                letter_part = citation_match.group(1)  # "- [B] "
                title = prefix_to_title.get(prefix)
                if title:
                    # Replace with: "- [B] Document Title" (newline will be added by join)
                    sources_lines_final.append(f"{letter_part}{title}")
                else:
                    # Keep original if title not found
                    sources_lines_final.append(line)
            else:
                # Keep non-citation lines as-is
                sources_lines_final.append(line)
        
        # Rebuild Sources section with each citation on its own line
        sources_section_replaced = "\n" + "\n".join(sources_lines_final)
        
        updated_answer = updated_answer.rstrip()
        # Add extra spacing before Sources section to make it more visible
        updated_answer += "\n\n" + sources_section_replaced
        logger.info(f"Added Sources section to final answer (with title replacements). Sources section length: {len(sources_section_replaced)}")
    elif pruned_citations:
        # Fallback: if no Sources section from LLM, use pruned_citations (old behavior)
        updated_answer = updated_answer.rstrip()
        updated_answer += "\n\nSources: " + ", ".join(pruned_citations)
        logger.info(f"Added fallback Sources section using pruned_citations")
    else:
        logger.warning("No Sources section to add - sources_section is empty and no pruned_citations available")
    
    # Add preserved "Documents used for analysis" section if we have one
    # This section MUST be preserved exactly as-is to maintain confidence scores
    if documents_analysis_section:
        updated_answer = updated_answer.rstrip()
        # Preserve the section exactly as extracted (don't strip too aggressively)
        # Only strip leading/trailing whitespace, preserve internal formatting
        documents_analysis_clean = documents_analysis_section.strip()
        updated_answer += "\n\n" + documents_analysis_clean
        # Verify contribution strength scores are still present after adding (check for both old "confidence" and new "contribution strength")
        has_contribution_after = '(contribution strength:' in documents_analysis_clean.lower() or 'contribution strength:' in documents_analysis_clean.lower()
        has_confidence_after = '(confidence:' in documents_analysis_clean.lower() or 'confidence:' in documents_analysis_clean.lower()
        has_scores_after = has_contribution_after or has_confidence_after
        logger.info(f"Added 'Documents used for analysis' section to final answer. Section length: {len(documents_analysis_clean)}, has_scores: {has_scores_after}")
        if not has_scores_after:
            logger.error("CRITICAL: 'Documents used for analysis' section added but contribution strength scores missing!")
            logger.error(f"Section content: {documents_analysis_clean[:500]}")
    else:
        logger.warning("No 'Documents used for analysis' section to add")
    
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
    logger.info(f"Final answer contains 'Sources:': {'Sources:' in updated_answer}")
    logger.info(f"Final answer contains 'Documents used for analysis': {'Documents used for analysis' in updated_answer}")
    if "Sources:" in updated_answer:
        sources_start = updated_answer.find("Sources:")
        logger.info(f"Sources section in final answer: {updated_answer[sources_start:sources_start+300]}...")
    if "Documents used for analysis" in updated_answer:
        docs_start = updated_answer.find("Documents used for analysis")
        # Log more of the section to verify confidence scores are present
        docs_section_preview = updated_answer[docs_start:docs_start+500]
        logger.info(f"'Documents used for analysis' section in final answer (length: {len(updated_answer) - docs_start}): {docs_section_preview}...")
        # Verify contribution strength scores are in the final answer (check for both old "confidence" and new "contribution strength")
        has_contribution_final = '(contribution strength:' in updated_answer.lower() or 'contribution strength:' in updated_answer.lower()
        has_confidence_final = '(confidence:' in updated_answer.lower() or 'confidence:' in updated_answer.lower()
        has_scores_final = has_contribution_final or has_confidence_final
        logger.info(f"Contribution strength scores present in final answer: {has_scores_final}")
    else:
        logger.warning("'Documents used for analysis' section NOT found in final answer!")
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

