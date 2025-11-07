"""
Refine retrieve node: Optional additional retrieve step driven by critic's refinements.
"""
import logging
from typing import List, Dict, Any
from inference.graph.state import GraphState
from inference.graph.agent_logger import get_agent_logger
from retrieval.retrieval import retrieve_hybrid

logger = logging.getLogger(__name__)
agent_log = get_agent_logger()


def node_refine_retrieve(state: GraphState) -> GraphState:
    """Optional additional retrieve step driven by critic's refinements."""
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Refine Retriever - Fetching additional chunks from refinements")
    logger.info("-" * 40)
    
    refinements = state.get("refinements", [])
    if not refinements:
        logger.info("No refinements provided, skipping refinement retrieval")
        logger.info("-" * 40)
        return {}
    
    logger.info(f"Refinement queries: {refinements}")
    doc_id = state.get('doc_id')
    cross_doc = state.get('cross_doc', False)
    doc_ids_found = set(state.get('doc_ids', []))
    hits_all: List[Dict[str, Any]] = []
    for rq in refinements:
        logger.info(f"Retrieving for: {rq}")
        hits = retrieve_hybrid(rq, k=6, k_lex=30, k_vec=30, doc_id=doc_id, cross_doc=cross_doc)
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
        text_preview = hit.get('text', '')[:150] if hit.get('text') else 'N/A'
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
        logger.info(f"Found {len(doc_ids_found)} document(s) in refinement retrieval: {[d + "..." for d in doc_ids_found]}")
    
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

