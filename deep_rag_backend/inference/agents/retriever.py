"""
Retriever agent: Fetches relevant chunks from the vector database.
"""
import logging
from inference.agents.state import State
from retrieval.retrieval import retrieve_hybrid
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def retriever_agent(state: State) -> State:
    """Retriever agent: Fetches relevant chunks from the vector database."""
    logger.info("-" * 40)
    logger.info("AGENT: Retriever - Fetching relevant chunks")
    logger.info("-" * 40)
    q = f"{state['question']}  {state['plan']}"
    doc_id = state.get('doc_id')
    cross_doc = state.get('cross_doc', False)
    logger.info(f"Query: {q}")
    if doc_id:
        logger.info(f"Filtering to document: {doc_id}...")
    if cross_doc:
        logger.info("Cross-document retrieval enabled")
    
    k: int = int(os.getenv('K_RETRIEVER', '8'))
    k_lex: int = int(os.getenv('K_LEX', '60'))
    k_vec: int = int(os.getenv('K_VEC', '60'))
    logger.info(f"Retrieval Agent Parameters: k={k}, k_lex={k_lex}, k_vec={k_vec}")

    hits = retrieve_hybrid(q, k, k_lex, k_vec, doc_id=doc_id, cross_doc=cross_doc)
    state["evidence"] = hits
    
    # Track all doc_ids from retrieved chunks
    doc_ids_found = set()
    for hit in hits:
        hit_doc_id = hit.get('doc_id')
        if hit_doc_id:
            doc_ids_found.add(hit_doc_id)
    
    if doc_ids_found:
        state["doc_ids"] = list(doc_ids_found)
        logger.info(f"Found {len(doc_ids_found)} document(s) in retrieved chunks: {[d + '...' for d in doc_ids_found]}")
    elif not state.get('doc_ids'):
        state["doc_ids"] = []
    
    logger.info(f"Retrieved {len(hits)} chunks:")
    for i, hit in enumerate(hits[:10], 1):  # Log top 10 for better visibility
        logger.info(f"  [{i}] Chunk ID: {hit.get('chunk_id', 'N/A')[:8]}...")
        logger.info(f"      Pages: {hit.get('p0', 'N/A')}-{hit.get('p1', 'N/A')}")
        logger.info(f"      Content Type: {hit.get('content_type', 'N/A')}")
        logger.info(f"      Scores: lex={hit.get('lex', 0):.4f}, vec={hit.get('vec', 0):.4f}, ce={hit.get('ce', 0):.4f}")
        # Show more text preview (200 chars) to understand what was retrieved
        text_preview = hit.get('text', '')[:200] if hit.get('text') else 'N/A'
        logger.info(f"      Text preview: {text_preview}...")
    if len(hits) > 10:
        logger.info(f"  ... and {len(hits) - 10} more chunks")
    # Log page distribution to see if all pages are represented
    pages_found = sorted(set([h.get('p0', 0) for h in hits]))
    logger.info(f"Pages represented in retrieved chunks: {pages_found}")
    logger.info("-" * 40)
    return state

