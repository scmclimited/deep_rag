"""
Stage 1: Primary retrieval from doc_id or all documents.
"""
import numpy as np
import logging
from typing import Optional, Union, List, Dict
from PIL import Image

from retrieval.db_utils import connect
from retrieval.sql import get_hybrid_sql
from retrieval.sanitize import sanitize_query_for_tsquery
from retrieval.vector_utils import parse_vector
from retrieval.reranker import rerank_candidates
from retrieval.mmr import mmr
from ingestion.embeddings import embed_text, embed_image, embed_multi_modal, EMBEDDING_DIM

logger = logging.getLogger(__name__)


def retrieve_stage_one(
    query: str,
    k: int,
    k_lex: int,
    k_vec: int,
    query_image: Optional[Union[str, Image.Image]],
    doc_id: Optional[str]
) -> List[Dict]:
    """
    Stage 1: Primary retrieval from doc_id or all documents.
    
    Args:
        query: Text query string
        k: Number of results to return
        k_lex: Number of lexical results to retrieve
        k_vec: Number of vector results to retrieve
        query_image: Optional image to combine with text query
        doc_id: Optional document ID to filter chunks
        
    Returns:
        List of retrieved chunks with scores
    """
    # Embed query using CLIP (text or text+image)
    if query_image:
        qemb = embed_multi_modal(text=query, image_path=query_image, normalize_emb=True)
    else:
        qemb = embed_text(query, normalize_emb=True)
    
    # Convert numpy array to list of Python floats for psycopg2/pgvector compatibility
    if isinstance(qemb, np.ndarray):
        qemb_list = qemb.astype(np.float64).tolist()
    else:
        qemb_list = [float(x) for x in qemb]
    
    # Ensure we have the expected number of dimensions
    if len(qemb_list) != EMBEDDING_DIM:
        raise ValueError(f"Expected embedding dimension {EMBEDDING_DIM}, got {len(qemb_list)}")
    
    # Sanitize query for tsquery
    sanitized_query = sanitize_query_for_tsquery(query)
    
    if sanitized_query != query:
        logger.debug(f"Query sanitized: '{query}' -> '{sanitized_query}'")
    
    # Generate SQL with optional doc_id filter
    hybrid_sql = get_hybrid_sql(EMBEDDING_DIM, doc_id=doc_id)
    
    if doc_id:
        logger.info(f"Filtering retrieval to document {doc_id}...")
    elif not doc_id:
        logger.info("Cross-document search enabled (no doc_id filter)")
    
    with connect() as conn, conn.cursor() as cur:
        try:
            params = {
                "q": query,
                "q_ts": sanitized_query,
                "emb": qemb_list,
                "k": k_lex + k_vec,
                "k_lex": k_lex,
                "k_vec": k_vec
            }
            if doc_id:
                params["doc_id"] = doc_id
            
            cur.execute(hybrid_sql, params)
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"SQL query failed: {e}", exc_info=True)
            conn.rollback()
            raise

    # Pull dense embeddings for MMR/rerank
    ids = [r[0] for r in rows]  # chunk_id
    if not ids: 
        return []
    
    with connect() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                "SELECT chunk_id, text, emb, COALESCE(content_type, 'text'), COALESCE(image_path, '') FROM chunks WHERE chunk_id = ANY(%s::uuid[])", 
                (ids,)
            )
            id2 = {cid: (txt, parse_vector(emb), content_type or 'text', image_path or '') 
                   for cid, txt, emb, content_type, image_path in cur.fetchall()}
        except Exception:
            cur.execute(
                "SELECT chunk_id, text, emb FROM chunks WHERE chunk_id = ANY(%s::uuid[])", 
                (ids,)
            )
            id2 = {cid: (txt, parse_vector(emb), 'text', '') 
                   for cid, txt, emb in cur.fetchall()}

    cands = []
    for r in rows:
        cid, doc_id, txt, p0, p1, content_type, image_path, lex_s, vec_s = r
        txt, emb, ct, img_path = id2.get(cid, (txt, None, content_type, image_path))
        if emb is None:
            continue
        cands.append({
            "chunk_id": cid,
            "doc_id": doc_id,
            "text": txt,
            "emb": emb,
            "p0": p0,
            "p1": p1,
            "content_type": ct or content_type,
            "image_path": img_path or image_path,
            "lex": float(lex_s),
            "vec": float(vec_s)
        })

    # Cross-encoder rerank (text-only, heavy precision step)
    cands = rerank_candidates(query, cands)

    # MMR diversify on top N, then return top-k
    diversified = mmr(cands[:30], qemb, lambda_mult=0.5, k=k)
    return diversified

