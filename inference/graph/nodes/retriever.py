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
    doc_id = state.get('doc_id')
    logger.info(f"Query: {q}")
    if doc_id:
        logger.info(f"Filtering to document: {doc_id}...")
    logger.info(f"Retrieval parameters: k=8, k_lex=40, k_vec=40")
    
    cross_doc = state.get('cross_doc', False)
    hits = retrieve_hybrid(q, k=8, k_lex=40, k_vec=40, doc_id=doc_id, cross_doc=cross_doc)
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
        logger.info(f"Found {len(doc_ids_found)} document(s) in retrieved chunks: {[d + "..." for d in doc_ids_found]}")
    
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
    if doc_ids_found:
        result["doc_ids"] = list(doc_ids_found)
    return result

