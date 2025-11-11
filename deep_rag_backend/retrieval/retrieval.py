"""
Main retrieval module - Hybrid retrieval with multi-modal support.

This module provides the main retrieve_hybrid function and maintains
backward compatibility by importing from modularized submodules.
"""
import logging
from typing import Optional, Union, List
from PIL import Image

# Import from modularized submodules
from retrieval.wait import wait_for_chunks
from retrieval.stages import retrieve_stage_one, retrieve_stage_two, merge_and_deduplicate
import os
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Export for backward compatibility
__all__ = [
    "retrieve_hybrid",
    "wait_for_chunks",
]


def retrieve_hybrid(
    query: str, 
    k: int = int(os.getenv("K_RETRIEVER", "6")),
    k_lex: int = int(os.getenv("K_LEX", "60")), 
    k_vec: int = int(os.getenv("K_VEC", "60")),
    query_image: Optional[Union[str, Image.Image]] = None,
    doc_id: Optional[str] = None,
    cross_doc: bool = False
) -> List[dict]:
    """
    Hybrid retrieval with multi-modal support and optional document filtering.
    
    Supports two-stage retrieval when cross_doc=True and doc_id is provided:
    1. First stage: Retrieve from doc_id (primary retrieval)
    2. Second stage: Embed query + retrieved content, then search semantically across all docs
    
    Args:
        query: Text query string
        k: Number of results to return
        k_lex: Number of lexical results to retrieve
        k_vec: Number of vector results to retrieve
        query_image: Optional image to combine with text query for multi-modal search
        doc_id: Optional document ID to filter chunks to a specific document
        cross_doc: If True and doc_id provided, perform two-stage retrieval (doc_id first, then cross-doc semantic search)
                   If True and doc_id not provided, enable cross-document search
        
    Returns:
        List of retrieved chunks with scores
    """
    # Two-stage retrieval when cross_doc=True and doc_id is provided
    if cross_doc and doc_id:
        logger.info(f"Two-stage retrieval: First stage from doc_id {doc_id}..., then cross-document semantic search")
        
        # Stage 1: Retrieve from doc_id (primary retrieval)
        primary_chunks = retrieve_stage_one(query, k, k_lex, k_vec, query_image, doc_id)
        
        if not primary_chunks:
            logger.warning(f"No chunks found for doc_id {doc_id[:8]}..., falling back to cross-document search")
            # Fall back to cross-document search if no primary chunks found
            return retrieve_stage_two(query, k, k_lex, k_vec, query_image, None)
        
        # Stage 2: Embed query + retrieved content, then search semantically across all docs
        # Combine query with retrieved content for better semantic search
        combined_text = query + " " + " ".join([c["text"][:500] for c in primary_chunks[:5]])  # Use top 5 chunks
        logger.info(f"Stage 2: Cross-document semantic search with combined query + retrieved content")
        
        secondary_chunks = retrieve_stage_two(combined_text, k, k_lex, k_vec, query_image, doc_id)
        
        # Merge and deduplicate results (prioritize primary chunks)
        all_chunks = merge_and_deduplicate(primary_chunks, secondary_chunks, k)
        
        return all_chunks
    
    # Standard single-stage retrieval
    # If cross_doc=True but no doc_id, enable cross-document search
    # If cross_doc=False and doc_id provided, strict doc_id filtering (ONLY search within doc_id)
    # If cross_doc=False and no doc_id, search all documents
    
    # CRITICAL FIX: When cross_doc=False and doc_id is provided, ONLY search within that doc_id
    # When cross_doc=True, allow cross-document search (pass None to search all)
    if cross_doc:
        # Cross-doc enabled: search all documents (pass None to retrieve_stage_one)
        return retrieve_stage_one(query, k, k_lex, k_vec, query_image, None)
    else:
        # Cross-doc disabled: if doc_id provided, ONLY search within that doc_id
        # If no doc_id, search all documents (fallback behavior)
        return retrieve_stage_one(query, k, k_lex, k_vec, query_image, doc_id)
