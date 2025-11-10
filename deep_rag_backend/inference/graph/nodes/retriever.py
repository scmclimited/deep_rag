"""
Retriever node: Fetches relevant chunks from the vector database.
"""
import logging
from inference.graph.state import GraphState
from inference.graph.agent_logger import get_agent_logger
from retrieval.retrieval import retrieve_hybrid

logger = logging.getLogger(__name__)
agent_log = get_agent_logger()


def node_retriever(state: GraphState) -> GraphState:
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Retriever - Fetching relevant chunks")
    logger.info(
        "State snapshot → iterations=%s, cross_doc=%s, selected_doc_ids=%s",
        state.get('iterations', 0),
        state.get('cross_doc', False),
        state.get('selected_doc_ids'),
    )
    logger.info("-" * 40)
    q = f"{state['question']}  {state.get('plan','')}"
    
    # Handle multi-document selection or single doc_id
    selected_doc_ids = state.get('selected_doc_ids')
    doc_id = state.get('doc_id')
    
    cross_doc = state.get('cross_doc', False)
    
    # CRITICAL: If cross_doc=False and selected_doc_ids is explicitly empty (user deselected all), return empty
    # Check this FIRST before determining doc_ids_to_filter
    if not cross_doc and selected_doc_ids is not None and len(selected_doc_ids) == 0:
        logger.info("No documents selected and cross_doc=False - returning empty results")
        result = {"evidence": []}
        result["doc_ids"] = []
        return result
    
    # Determine which doc_ids to use for filtering
    # Priority: selected_doc_ids (explicit user selection) > doc_id (from ingestion/previous query)
    # If both are provided, combine them (user may have selected other docs in addition to ingested doc)
    doc_ids_to_filter = None
    if selected_doc_ids and len(selected_doc_ids) > 0:
        # User explicitly selected documents
        doc_ids_to_filter = list(selected_doc_ids)  # Make a copy to avoid modifying original
        
        # If doc_id is also provided and not already in selected_doc_ids, add it
        # This handles the case where user ingested a doc AND selected other docs
        if doc_id and doc_id not in doc_ids_to_filter:
            doc_ids_to_filter.append(doc_id)
            logger.info(f"Combining selected_doc_ids with doc_id: {len(doc_ids_to_filter)} document(s) total")
        
        if len(doc_ids_to_filter) > 1:
            logger.info(f"Multi-document selection: {len(doc_ids_to_filter)} document(s)")
        else:
            logger.info(f"Filtering to document: {doc_ids_to_filter[0]}...")
    elif doc_id:
        # Fallback to doc_id if selected_doc_ids not provided
        doc_ids_to_filter = [doc_id]
        logger.info(f"Filtering to document: {doc_id}...")
    
    # CRITICAL: If cross_doc=False and no doc_ids_to_filter (no documents specified), return empty
    # Only search all documents when cross_doc=True
    if not cross_doc and doc_ids_to_filter is None:
        logger.info("No documents specified and cross_doc=False - returning empty results")
        result = {"evidence": []}
        result["doc_ids"] = []
        return result
    
    logger.info(f"Retrieval parameters: k=20, k_lex=100, k_vec=100")
    
    # HYBRID APPROACH: Enhanced cross-doc + selection handling
    # When cross_doc=True AND selected_doc_ids provided:
    # - Prioritize selected documents (retrieve more from them)
    # - But still allow cross-doc retrieval for supplementary context
    if cross_doc and doc_ids_to_filter and len(doc_ids_to_filter) > 0:
        logger.info(f"HYBRID MODE: Cross-doc enabled with {len(doc_ids_to_filter)} selected document(s)")
        logger.info("  Strategy: Prioritize selected docs, supplement with cross-doc if needed")
        
        # First, retrieve from selected documents (higher k for better coverage)
        selected_hits = []
        for selected_doc in doc_ids_to_filter:
            logger.info(f"  Retrieving from selected document: {selected_doc[:8]}...")
            doc_hits = retrieve_hybrid(q, k=15, k_lex=75, k_vec=75, doc_id=selected_doc, cross_doc=False)
            selected_hits.extend(doc_hits)
            logger.info(f"    Found {len(doc_hits)} chunks")
        
        # Remove duplicates from selected hits
        seen_selected = set()
        unique_selected_hits = []
        for h in selected_hits:
            if h["chunk_id"] not in seen_selected:
                seen_selected.add(h["chunk_id"])
                unique_selected_hits.append(h)
        
        logger.info(f"  Total from selected documents: {len(unique_selected_hits)} unique chunks")
        
        # If we have good coverage from selected docs, use them
        # Otherwise, supplement with cross-doc retrieval
        if len(unique_selected_hits) >= 12:
            logger.info("  Sufficient coverage from selected documents - using them")
            hits = unique_selected_hits[:20]  # Cap at 20 for consistency
        else:
            logger.info(f"  Limited coverage ({len(unique_selected_hits)} chunks) - supplementing with cross-doc")
            # Retrieve from all documents to supplement
            cross_doc_hits = retrieve_hybrid(q, k=20, k_lex=100, k_vec=100, doc_id=None, cross_doc=True)
            
            # Merge selected hits (prioritized) with cross-doc hits
            seen_all = set()
            merged_hits = []
            
            # Add selected hits first (highest priority)
            for h in unique_selected_hits:
                if h["chunk_id"] not in seen_all:
                    seen_all.add(h["chunk_id"])
                    merged_hits.append(h)
            
            # Add cross-doc hits to fill gaps
            for h in cross_doc_hits:
                if h["chunk_id"] not in seen_all and len(merged_hits) < 20:
                    seen_all.add(h["chunk_id"])
                    merged_hits.append(h)
            
            hits = merged_hits
            logger.info(f"  Merged result: {len(hits)} chunks ({len(unique_selected_hits)} from selected, {len(hits) - len(unique_selected_hits)} from cross-doc)")
    else:
        # Standard retrieval (no hybrid mode)
        # CRITICAL FIX: For multi-document queries, retrieve from ALL selected documents
        if not cross_doc and doc_ids_to_filter and len(doc_ids_to_filter) > 1:
            # Multi-document query without cross-doc: retrieve from each document
            logger.info(f"Multi-document retrieval: fetching from {len(doc_ids_to_filter)} document(s)")
            all_hits = []
            for doc in doc_ids_to_filter:
                logger.info(f"  Retrieving from document: {doc[:8]}...")
                doc_hits = retrieve_hybrid(q, k=15, k_lex=75, k_vec=75, doc_id=doc, cross_doc=False)
                all_hits.extend(doc_hits)
                logger.info(f"    Found {len(doc_hits)} chunks")
            
            # Remove duplicates
            seen = set()
            hits = []
            for h in all_hits:
                if h["chunk_id"] not in seen:
                    seen.add(h["chunk_id"])
                    hits.append(h)
            
            logger.info(f"  Total: {len(hits)} unique chunks from {len(doc_ids_to_filter)} documents")
        else:
            # Single document or cross-doc query
            # CRITICAL FIX: If specific documents are selected, query those documents (ignore cross_doc flag)
            # cross_doc flag only applies when no specific documents are selected
            if doc_ids_to_filter and len(doc_ids_to_filter) > 0:
                # User selected specific document(s) - query those documents, not cross-doc
                doc_id_for_retrieval = doc_ids_to_filter[0]
                cross_doc_for_retrieval = False  # Force single-doc when specific doc is selected
                logger.info(f"Querying specific document: {doc_id_for_retrieval[:8]}... (cross_doc flag ignored)")
            else:
                # No specific documents selected - use cross_doc flag
                doc_id_for_retrieval = None
                cross_doc_for_retrieval = cross_doc
                if cross_doc:
                    logger.info("Cross-document search enabled (no specific documents selected)")
            
            hits = retrieve_hybrid(q, k=20, k_lex=100, k_vec=100, doc_id=doc_id_for_retrieval, cross_doc=cross_doc_for_retrieval)
            
            # Filter hits to only include selected doc_ids if specific documents were selected
            if doc_ids_to_filter and len(doc_ids_to_filter) > 0:
                doc_ids_set = set(doc_ids_to_filter)
                hits = [h for h in hits if h.get('doc_id') in doc_ids_set]
                logger.info(f"Filtered to {len(hits)} chunks from selected documents")
    # Merge with any prior evidence (e.g., from refinement loops)
    seen, merged = set(), []
    for h in (state.get("evidence", []) + hits):
        if h["chunk_id"] in seen:
            continue
        seen.add(h["chunk_id"]); merged.append(h)
    
    # Track all doc_ids from retrieved chunks
    doc_ids_found = set()
    for hit in merged:
        hit_doc_id = hit.get('doc_id')
        if hit_doc_id:
            doc_ids_found.add(hit_doc_id)
    
    if doc_ids_found:
        logger.info(f"Found {len(doc_ids_found)} document(s) in retrieved chunks: {[d + '...' for d in doc_ids_found]}")
    
    logger.info(f"Retrieved {len(hits)} new chunks, {len(merged)} total after merge")
    for i, hit in enumerate(merged[:10], 1):  # Log top 10 for better visibility
        logger.info(f"  [{i}] Chunk ID: {hit.get('chunk_id', 'N/A')[:8]}...")
        logger.info(f"      Pages: {hit.get('p0', 'N/A')}-{hit.get('p1', 'N/A')}")
        logger.info(f"      Content Type: {hit.get('content_type', 'N/A')}")
        logger.info(f"      Scores: lex={hit.get('lex', 0):.4f}, vec={hit.get('vec', 0):.4f}, ce={hit.get('ce', 0):.4f}")
        # Show text preview (first 200 chars) to understand what was retrieved
        text_preview = hit.get('text', '')[:200] if hit.get('text') else 'N/A'
        logger.info(f"      Text preview: {text_preview}...")
    if len(merged) > 10:
        logger.info(f"  ... and {len(merged) - 10} more chunks")
    # Log page distribution to see if all pages are represented
    pages_found = sorted(set([h.get('p0', 0) for h in merged]))
    logger.info(f"Pages represented in retrieved chunks: {pages_found}")
    logger.info("-" * 40)
    
    # Log to agent logger with detailed retrieval info
    agent_log.log_step(
        node="retriever",
        action="retrieve",
        query=q,
        num_chunks=len(merged),
        pages=pages_found,
        metadata={
            "new_chunks": len(hits),
            "total_chunks": len(merged),
            "top_scores": [
                {
                    "lex": h.get('lex', 0),
                    "vec": h.get('vec', 0),
                    "ce": h.get('ce', 0)
                } for h in merged[:5]
            ]
        }
    )
    
    # Log detailed retrieval results for analysis
    agent_log.log_retrieval_details(
        session_id="current",
        query=q,
        chunks=merged
    )
    
    result = {"evidence": merged}
    # Always include doc_ids, even if empty
    result["doc_ids"] = list(doc_ids_found) if doc_ids_found else []
    return result

