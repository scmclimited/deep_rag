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
    
    logger.info(f"Retrieval parameters: k=8, k_lex=40, k_vec=40")
    
    # For now, use first doc_id for backward compatibility (will enhance retrieve_hybrid later)
    doc_id_for_retrieval = doc_ids_to_filter[0] if doc_ids_to_filter and not cross_doc else None
    
    hits = retrieve_hybrid(q, k=8, k_lex=40, k_vec=40, doc_id=doc_id_for_retrieval, cross_doc=cross_doc)
    
    # Filter hits to only include selected doc_ids if cross_doc=False and selected_doc_ids provided
    if not cross_doc and doc_ids_to_filter and len(doc_ids_to_filter) > 0:
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

