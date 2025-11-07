"""
Stage 2: Cross-document semantic search (excluding primary doc_id if provided).
"""
import numpy as np
import logging
from typing import Optional, Union, List, Dict
from PIL import Image

from retrieval.db_utils import connect
from retrieval.sql import get_hybrid_sql_with_exclusion
from retrieval.sanitize import sanitize_query_for_tsquery
from retrieval.vector_utils import parse_vector
from retrieval.reranker import rerank_candidates
from retrieval.mmr import mmr
from ingestion.embeddings import embed_text, embed_image, embed_multi_modal, EMBEDDING_DIM

logger = logging.getLogger(__name__)


def retrieve_stage_two(
    query: str,
    k: int,
    k_lex: int,
    k_vec: int,
    query_image: Optional[Union[str, Image.Image]],
    exclude_doc_id: Optional[str]
) -> List[Dict]:
    """
    Stage 2: Cross-document semantic search (excluding primary doc_id if provided).
    
    Args:
        query: Text query string
        k: Number of results to return
        k_lex: Number of lexical results to retrieve
        k_vec: Number of vector results to retrieve
        query_image: Optional image to combine with text query
        exclude_doc_id: Optional document ID to exclude from results
        
    Returns:
        List of retrieved chunks with scores
    """
    # Embed query using CLIP (text or text+image)
    if query_image:
        qemb = embed_multi_modal(text=query, image_path=query_image, normalize_emb=True)
    else:
        qemb = embed_text(query, normalize_emb=True)
    
    # Convert numpy array to list of Python floats
    if isinstance(qemb, np.ndarray):
        qemb_list = qemb.astype(np.float64).tolist()
    else:
        qemb_list = [float(x) for x in qemb]
    
    if len(qemb_list) != EMBEDDING_DIM:
        raise ValueError(f"Expected embedding dimension {EMBEDDING_DIM}, got {len(qemb_list)}")
    
    # Sanitize query for tsquery
    sanitized_query = sanitize_query_for_tsquery(query)
    
    # Generate SQL with exclusion filter for primary doc_id
    hybrid_sql = get_hybrid_sql_with_exclusion(EMBEDDING_DIM, exclude_doc_id=exclude_doc_id)
    
    if exclude_doc_id:
        logger.info(f"Cross-document search: Excluding primary doc_id {exclude_doc_id[:8]}...")
    else:
        logger.info("Cross-document search: Searching all documents")
    
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
            if exclude_doc_id:
                params["exclude_doc_id"] = exclude_doc_id
            
            cur.execute(hybrid_sql, params)
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"SQL query failed: {e}", exc_info=True)
            conn.rollback()
            raise

    ids = [r[0] for r in rows]
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

    # Cross-encoder rerank
    cands = rerank_candidates(query, cands)

    # MMR diversify
    diversified = mmr(cands[:30], qemb, lambda_mult=0.5, k=k)
    return diversified

