"""
Retriever node: Fetches relevant chunks from the vector database.
"""
import logging
from inference.graph.state import GraphState
from inference.graph.agent_logger import get_agent_logger
from retrieval.retrieval import retrieve_hybrid
from retrieval.document_structure import retrieve_by_document_structure
import os

from dotenv import load_dotenv
load_dotenv()

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
    
    # Handle multi-document selection, uploaded documents, or single doc_id
    selected_doc_ids = state.get('selected_doc_ids')
    uploaded_doc_ids = state.get('uploaded_doc_ids')
    doc_id = state.get('doc_id')
    
    cross_doc = state.get('cross_doc', False)
    
    # When uploaded_doc_ids is present:
    # - If cross_doc=False: Scope to ONLY attached documents (like selected documents)
    # - If cross_doc=True: Prioritize attached documents but allow cross-doc retrieval (HYBRID MODE)
    # We don't force cross_doc=False here - we respect the user's cross_doc setting
    if uploaded_doc_ids and len(uploaded_doc_ids) > 0:
        if cross_doc:
            logger.info(f"🔄 uploaded_doc_ids present ({len(uploaded_doc_ids)} document(s)) with cross_doc=True - will prioritize attached docs but allow cross-doc")
        else:
            logger.info(f"🔒 uploaded_doc_ids present ({len(uploaded_doc_ids)} document(s)) with cross_doc=False - will scope to ONLY attached documents")
    
    # CRITICAL: If cross_doc=False and selected_doc_ids is explicitly empty (user deselected all), return empty
    # Check this FIRST before determining doc_ids_to_filter
    if not cross_doc and selected_doc_ids is not None and len(selected_doc_ids) == 0:
        logger.info("No documents selected and cross_doc=False - returning empty results")
        result = {"evidence": []}
        result["doc_ids"] = []
        return result
    
    # Determine which doc_ids to use for filtering
    # Priority: selected_doc_ids (explicit user selection) > uploaded_doc_ids (attached docs) > doc_id (from ingestion/previous query)
    # Combine all provided document IDs (user may have selected, attached, and ingested docs)
    doc_ids_to_filter = None
    
    # Start with selected_doc_ids if provided
    if selected_doc_ids and len(selected_doc_ids) > 0:
        doc_ids_to_filter = list(selected_doc_ids)  # Make a copy to avoid modifying original
        logger.info(f"Starting with {len(doc_ids_to_filter)} selected document(s)")
    
    # Add uploaded_doc_ids if provided (attached documents)
    if uploaded_doc_ids and len(uploaded_doc_ids) > 0:
        if doc_ids_to_filter is None:
            doc_ids_to_filter = []
        for uploaded_id in uploaded_doc_ids:
            if uploaded_id not in doc_ids_to_filter:
                doc_ids_to_filter.append(uploaded_id)
        logger.info(f"Added {len(uploaded_doc_ids)} uploaded document(s), total: {len(doc_ids_to_filter)} document(s)")
    
    # Add doc_id if provided and not already included
    if doc_id:
        if doc_ids_to_filter is None:
            doc_ids_to_filter = [doc_id]
            logger.info(f"Using doc_id: {doc_id[:8]}...")
        elif doc_id not in doc_ids_to_filter:
            doc_ids_to_filter.append(doc_id)
            logger.info(f"Combining with doc_id: {len(doc_ids_to_filter)} document(s) total")
    
    if doc_ids_to_filter:
        if len(doc_ids_to_filter) > 1:
            logger.info(f"Multi-document selection: {len(doc_ids_to_filter)} document(s)")
        else:
            logger.info(f"Filtering to document: {doc_ids_to_filter[0][:8]}...")
    
    # CRITICAL: If cross_doc=False and no doc_ids_to_filter (no documents specified), return empty
    # Only search all documents when cross_doc=True
    if not cross_doc and doc_ids_to_filter is None:
        logger.info("No documents specified and cross_doc=False - returning empty results")
        result = {"evidence": []}
        result["doc_ids"] = []
        return result
    
    logger.info(f"Retrieval parameters: k={os.getenv('K_RETRIEVER', '10')}, k_lex={os.getenv('K_LEX', '60')}, k_vec={os.getenv('K_VEC', '60')}")
    k = int(os.getenv('K_RETRIEVER', '8'))
    k_lex = int(os.getenv('K_LEX', '60'))
    k_vec = int(os.getenv('K_VEC', '60'))
    
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
            doc_hits = retrieve_hybrid(q, k, k_lex, k_vec, doc_id=selected_doc, cross_doc=False)
            selected_hits.extend(doc_hits)
            logger.info(f"    Found {len(doc_hits)} chunks via similarity search")
            
            # Check if similarity is poor and supplement with structure-based retrieval
            has_good_similarity = any(
                h.get("ce", 0) > 0.3 or
                (h.get("lex", 0) > 0 and h.get("vec", 0) > 0.6) or
                h.get("vec", 0) > 0.7
                for h in doc_hits
            )
            
            logger.info(f"    Similarity check: has_good_similarity={has_good_similarity}, "
                       f"top_scores: ce={max((h.get('ce', 0) for h in doc_hits), default=0):.3f}, "
                       f"vec={max((h.get('vec', 0) for h in doc_hits), default=0):.3f}, "
                       f"lex={max((h.get('lex', 0) for h in doc_hits), default=0):.3f}")
            
            if not has_good_similarity:
                logger.info(f"    Similarity results poor - supplementing with structure-based retrieval")
                structure_hits = retrieve_by_document_structure(
                    doc_id=selected_doc,
                    max_chunks=15,
                    strategy="first_pages"
                )
                
                # Merge structure hits with similarity hits (deduplicate by chunk_id)
                seen_chunk_ids = {h["chunk_id"] for h in doc_hits}
                for struct_hit in structure_hits:
                    if struct_hit["chunk_id"] not in seen_chunk_ids:
                        selected_hits.append(struct_hit)
                        seen_chunk_ids.add(struct_hit["chunk_id"])
                        logger.debug(f"      Added structure chunk: page {struct_hit.get('p0')}")
                
                logger.info(f"    Total after structure supplement: {len([h for h in selected_hits if h.get('doc_id') == selected_doc])} chunks")
        
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
            cross_doc_hits = retrieve_hybrid(q, k, k_lex, k_vec, doc_id=None, cross_doc=True)
            
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
    elif doc_ids_to_filter and len(doc_ids_to_filter) > 0:
        # Force retrieval strictly within selected documents when cross_doc=False
        logger.info("Selective retrieval mode: restricting search to explicitly selected documents")
        all_hits = []
        for doc in doc_ids_to_filter:
            logger.info(f"  Retrieving from selected document: {doc[:8]}...")
            doc_hits = retrieve_hybrid(q, k, k_lex, k_vec, doc_id=doc, cross_doc=False)
            all_hits.extend(doc_hits)
            logger.info(f"    Found {len(doc_hits)} chunks via similarity search")
            
            # ENHANCEMENT: For explicit document selection with ambiguous queries,
            # supplement with structure-based retrieval if similarity results are poor
            # Check if we have good similarity scores - require stronger signals to avoid false positives
            # A good match requires:
            #   - Positive CE score (> 0.3) OR
            #   - Both lexical AND vector match (lex > 0 AND vec > 0.6) OR  
            #   - Very high vector score alone (> 0.7)
            # This prevents false positives from marginal vector matches (e.g., 0.607) with negative CE scores
            has_good_similarity = any(
                h.get("ce", 0) > 0.3 or  # Good cross-encoder score
                (h.get("lex", 0) > 0 and h.get("vec", 0) > 0.6) or  # Both lexical and vector match
                h.get("vec", 0) > 0.7  # Very high vector score alone
                for h in doc_hits
            )
            
            # Log similarity check for debugging
            logger.info(f"    Similarity check: has_good_similarity={has_good_similarity}, "
                       f"top_scores: ce={max((h.get('ce', 0) for h in doc_hits), default=0):.3f}, "
                       f"vec={max((h.get('vec', 0) for h in doc_hits), default=0):.3f}, "
                       f"lex={max((h.get('lex', 0) for h in doc_hits), default=0):.3f}")
            
            # If similarity is poor, supplement with structure-based retrieval
            # Changed condition: trigger structure-based retrieval if similarity is poor, regardless of chunk count
            # This is critical for ambiguous queries like "share details about this document"
            # where similarity search may return many chunks but with poor scores
            if not has_good_similarity:
                logger.info(f"    Similarity results poor (has_good_similarity=False) - supplementing with structure-based retrieval")
                structure_hits = retrieve_by_document_structure(
                    doc_id=doc,
                    max_chunks=15,  # Get more chunks for document analysis
                    strategy="first_pages"  # Start with first pages for overview
                )
                
                # Merge structure hits with similarity hits (deduplicate by chunk_id)
                seen_chunk_ids = {h["chunk_id"] for h in doc_hits}
                for struct_hit in structure_hits:
                    if struct_hit["chunk_id"] not in seen_chunk_ids:
                        all_hits.append(struct_hit)
                        seen_chunk_ids.add(struct_hit["chunk_id"])
                        logger.debug(f"      Added structure chunk: page {struct_hit.get('p0')}")
                
                logger.info(f"    Total after structure supplement: {len([h for h in all_hits if h.get('doc_id') == doc])} chunks")

        # Deduplicate chunk hits and filter to only selected documents (safety check)
        seen = set()
        hits = []
        doc_ids_set = set(doc_ids_to_filter)
        for h in all_hits:
            if h["chunk_id"] not in seen:
                hit_doc_id = h.get('doc_id')
                # Only include chunks from selected documents
                if hit_doc_id and hit_doc_id in doc_ids_set:
                    seen.add(h["chunk_id"])
                    hits.append(h)
        logger.info(f"  Total restricted hits: {len(hits)} chunks from {len(doc_ids_to_filter)} documents")
    else:
        # Single document or cross-doc query without explicit selection
        doc_id_for_retrieval = None
        cross_doc_for_retrieval = cross_doc
        if cross_doc:
            logger.info("Cross-document search enabled (no specific documents selected)")

        hits = retrieve_hybrid(q, k=20, k_lex=100, k_vec=100, doc_id=doc_id_for_retrieval, cross_doc=cross_doc_for_retrieval)

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

