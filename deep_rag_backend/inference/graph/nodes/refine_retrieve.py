"""
Refine retrieve node: Optional additional retrieve step driven by critic's refinements.
"""
import os
import logging
from typing import List, Dict, Any
from inference.graph.state import GraphState
from inference.graph.agent_logger import get_agent_logger
from retrieval.retrieval import retrieve_hybrid
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
agent_log = get_agent_logger()


def node_refine_retrieve(state: GraphState) -> GraphState:
    """Optional additional retrieve step driven by critic's refinements."""
    logger.info("=" * 80)
    logger.info("GRAPH NODE: Refine Retriever - Fetching additional chunks from refinements")
    logger.info("=" * 80)
    logger.info(f"State snapshot:")
    logger.info(f"  - Iterations: {state.get('iterations', 0)}")
    logger.info(f"  - Pending refinements: {len(state.get('refinements', []))}")
    logger.info(f"  - Current evidence: {len(state.get('evidence', []))} chunks")
    logger.info(f"  - Cross-doc: {state.get('cross_doc', False)}")
    logger.info("-" * 80)
    k: int = int(os.getenv('K_RETRIEVER', '12'))
    k_lex: int = int(os.getenv('K_LEX', '72'))
    k_vec: int = int(os.getenv('K_VEC', '72'))
    logger.info(f"Refine Retrieval Parameters: k={k}, k_lex={k_lex}, k_vec={k_vec}")
    
    refinements = state.get("refinements", [])
    if not refinements:
        logger.info("No refinements provided, skipping refinement retrieval")
        logger.info("-" * 80)
        return {}
    
    logger.info(f"Processing {len(refinements)} refinement queries:")
    for i, ref in enumerate(refinements, 1):
        logger.info(f"  {i}. {ref}")
    
    doc_id = state.get('doc_id')
    selected_doc_ids = state.get('selected_doc_ids')
    cross_doc = state.get('cross_doc', False)
    uploaded_doc_ids = state.get('uploaded_doc_ids')
    doc_ids_found = set(state.get('doc_ids', []))
    hits_all: List[Dict[str, Any]] = []
    
    # When uploaded_doc_ids is present:
    # - If cross_doc=False: Scope to ONLY attached documents (like selected documents)
    # - If cross_doc=True: Prioritize attached documents but allow cross-doc retrieval
    # We don't force cross_doc=False here - we respect the user's cross_doc setting
    if uploaded_doc_ids and len(uploaded_doc_ids) > 0:
        if cross_doc:
            logger.info(f"🔄 uploaded_doc_ids present in refine_retrieve ({len(uploaded_doc_ids)} document(s)) with cross_doc=True - will prioritize attached docs but allow cross-doc")
        else:
            logger.info(f"🔒 uploaded_doc_ids present in refine_retrieve ({len(uploaded_doc_ids)} document(s)) with cross_doc=False - will scope to ONLY attached documents")
    
    # CRITICAL FIX: If specific documents are selected or uploaded, query those documents (ignore cross_doc flag)
    # cross_doc flag only applies when no specific documents are selected or uploaded
    doc_ids_to_filter = None
    if selected_doc_ids and len(selected_doc_ids) > 0:
        doc_ids_to_filter = list(selected_doc_ids)
        # Add uploaded_doc_ids if provided
        if uploaded_doc_ids and len(uploaded_doc_ids) > 0:
            for uploaded_id in uploaded_doc_ids:
                if uploaded_id not in doc_ids_to_filter:
                    doc_ids_to_filter.append(uploaded_id)
        if doc_id and doc_id not in doc_ids_to_filter:
            doc_ids_to_filter.append(doc_id)
    elif uploaded_doc_ids and len(uploaded_doc_ids) > 0:
        # Only uploaded_doc_ids provided (no selected_doc_ids)
        doc_ids_to_filter = list(uploaded_doc_ids)
        if doc_id and doc_id not in doc_ids_to_filter:
            doc_ids_to_filter.append(doc_id)
    elif doc_id:
        doc_ids_to_filter = [doc_id]
    
    # Determine doc_id and cross_doc for retrieval
    # When specific documents are selected/uploaded:
    # - If cross_doc=False: Query only those documents
    # - If cross_doc=True: Query those documents first, then supplement with cross-doc if needed
    if doc_ids_to_filter and len(doc_ids_to_filter) > 0:
        if cross_doc:
            # HYBRID MODE: Prioritize selected/uploaded docs but allow cross-doc
            if len(doc_ids_to_filter) > 1:
                logger.info(f"Refinement queries will prioritize {len(doc_ids_to_filter)} specific document(s) with cross-doc enabled")
            else:
                logger.info(f"Refinement queries will prioritize specific document: {doc_ids_to_filter[0][:8]}... with cross-doc enabled")
        else:
            # Scoped mode: Only query selected/uploaded documents
            if len(doc_ids_to_filter) > 1:
                logger.info(f"Refinement queries will target {len(doc_ids_to_filter)} specific document(s) (cross_doc disabled)")
            else:
                logger.info(f"Refinement queries will target specific document: {doc_ids_to_filter[0][:8]}... (cross_doc disabled)")
    else:
        doc_id_for_retrieval = None
        cross_doc_for_retrieval = cross_doc
        if cross_doc:
            logger.info("Refinement queries will use cross-document search (no specific documents selected)")
    
    for idx, rq in enumerate(refinements, 1):
        logger.info(f"Refinement {idx}/{len(refinements)}: {rq}")
        # If specific documents are selected/uploaded
        if doc_ids_to_filter and len(doc_ids_to_filter) > 0:
            hits = []
            # First, retrieve from selected/uploaded documents
            for doc_id_for_retrieval in doc_ids_to_filter:
                doc_hits = retrieve_hybrid(rq, k, k_lex, k_vec, doc_id=doc_id_for_retrieval, cross_doc=False)
                hits.extend(doc_hits)
                logger.info(f"  Retrieved {len(doc_hits)} chunks from document: {doc_id_for_retrieval[:8]}...")
            
            # If cross_doc=True and we have limited coverage, supplement with cross-doc retrieval
            if cross_doc and len(hits) < 12:
                logger.info(f"  Limited coverage ({len(hits)} chunks) - supplementing with cross-doc retrieval")
                cross_doc_hits = retrieve_hybrid(rq, k, k_lex, k_vec, doc_id=None, cross_doc=True)
                # Filter to exclude chunks from already-retrieved documents
                doc_ids_set = set(doc_ids_to_filter)
                cross_doc_hits_filtered = [h for h in cross_doc_hits if h.get('doc_id') not in doc_ids_set]
                hits.extend(cross_doc_hits_filtered)
                logger.info(f"  Added {len(cross_doc_hits_filtered)} chunks from cross-doc retrieval")
        else:
            hits = retrieve_hybrid(rq, k, k_lex, k_vec, doc_id=None, cross_doc=cross_doc)
        
        # Filter hits based on cross_doc setting
        if doc_ids_to_filter and len(doc_ids_to_filter) > 0:
            if cross_doc:
                # cross_doc=True: Allow hits from selected/uploaded docs AND cross-doc hits
                # (hits already include both from the logic above)
                logger.info(f"  Retrieved {len(hits)} chunks (prioritized from selected/uploaded docs, supplemented with cross-doc)")
            else:
                # cross_doc=False: Only allow hits from selected/uploaded documents
                doc_ids_set = set(doc_ids_to_filter)
                hits = [h for h in hits if h.get('doc_id') in doc_ids_set]
                logger.info(f"  Retrieved {len(hits)} chunks (filtered to selected/uploaded documents only)")
        else:
            logger.info(f"  Retrieved {len(hits)} chunks")
        
        hits_all.extend(hits)
        
        # Track doc_ids from refinement retrieval
        for hit in hits:
            hit_doc_id = hit.get('doc_id')
            if hit_doc_id:
                doc_ids_found.add(hit_doc_id)
        
        # Log each refinement query
        agent_log.log_step(
            node="refine_retrieve",
            action="refine_query",
            query=rq,
            num_chunks=len(hits),
            pages=sorted(set([h.get('p0', 0) for h in hits]))
        )
    
    logger.info(f"Retrieved {len(hits_all)} additional chunks from refinements")
    
    # Log retrieved chunks with text preview
    for i, hit in enumerate(hits_all[:5], 1):
        logger.info(f"  Refinement [{i}] Pages: {hit.get('p0', 'N/A')}-{hit.get('p1', 'N/A')}")
        text_preview = hit.get('text', '')[:250] if hit.get('text') else 'N/A'
        logger.info(f"      Text preview: {text_preview}...")
    
    # Merge with existing evidence
    seen, merged = set(), []
    for h in (state.get("evidence", []) + hits_all):
        if h["chunk_id"] in seen:
            continue
        seen.add(h["chunk_id"]); merged.append(h)
    
    logger.info(f"Total evidence after merge: {len(merged)} chunks")
    
    # Update doc_ids in state
    if doc_ids_found:
        logger.info(f"Found {len(doc_ids_found)} document(s) in refinement retrieval: {[d + '...' for d in doc_ids_found]}")
    
    # Log page distribution after merge
    pages_found = sorted(set([h.get('p0', 0) for h in merged]))
    logger.info(f"Pages represented after merge: {pages_found}")
    logger.info("Routing back to compressor for re-compression")
    logger.info("-" * 40)
    
    # Log refinement retrieval summary
    agent_log.log_step(
        node="refine_retrieve",
        action="merge_results",
        num_chunks=len(merged),
        pages=pages_found,
        metadata={
            "refinement_chunks": len(hits_all),
            "total_after_merge": len(merged)
        }
    )
    
    result = {"evidence": merged}
    if doc_ids_found:
        result["doc_ids"] = list(doc_ids_found)
    return result

