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
    doc_ids_found = set(state.get('doc_ids', []))
    hits_all: List[Dict[str, Any]] = []
    
    # CRITICAL FIX: If specific documents are selected, query those documents (ignore cross_doc flag)
    # cross_doc flag only applies when no specific documents are selected
    doc_ids_to_filter = None
    if selected_doc_ids and len(selected_doc_ids) > 0:
        doc_ids_to_filter = list(selected_doc_ids)
        if doc_id and doc_id not in doc_ids_to_filter:
            doc_ids_to_filter.append(doc_id)
    elif doc_id:
        doc_ids_to_filter = [doc_id]
    
    # Determine doc_id and cross_doc for retrieval
    if doc_ids_to_filter and len(doc_ids_to_filter) > 0:
        doc_id_for_retrieval = doc_ids_to_filter[0]
        cross_doc_for_retrieval = False  # Force single-doc when specific doc is selected
        logger.info(f"Refinement queries will target specific document: {doc_id_for_retrieval[:8]}... (cross_doc flag ignored)")
    else:
        doc_id_for_retrieval = None
        cross_doc_for_retrieval = cross_doc
        if cross_doc:
            logger.info("Refinement queries will use cross-document search (no specific documents selected)")
    
    for idx, rq in enumerate(refinements, 1):
        logger.info(f"Refinement {idx}/{len(refinements)}: {rq}")
        hits = retrieve_hybrid(rq, k, k_lex, k_vec, doc_id=doc_id_for_retrieval, cross_doc=cross_doc_for_retrieval)
        
        # Filter hits to only include selected doc_ids if specific documents were selected
        if doc_ids_to_filter and len(doc_ids_to_filter) > 0:
            doc_ids_set = set(doc_ids_to_filter)
            hits = [h for h in hits if h.get('doc_id') in doc_ids_set]
            logger.info(f"  Retrieved {len(hits)} chunks (filtered to selected documents)")
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

