# retrieve.py
import psycopg2, numpy as np
import os
import re
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from typing import Optional, Union
from PIL import Image

load_dotenv()

# Use unified multi-modal embedding system (CLIP)
from ingestion.embeddings import embed_text, embed_image, embed_multi_modal, normalize, EMBEDDING_DIM

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Reranker for query time (text-only cross-encoder)
RERANK_MODEL = "BAAI/bge-reranker-base"
try:
    reranker = CrossEncoder(RERANK_MODEL)
except Exception as e:
    logger.warning(f"Reranker not available: {e}. Continuing without reranking.")
    reranker = None

def connect():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        dbname=os.getenv("DB_NAME")
    )

# Updated SQL to handle multi-modal embeddings (512 dims for CLIP)
# Also includes content_type and image_path fields
HYBRID_SQL = """
WITH
q AS (
  SELECT
    to_tsvector('simple', unaccent(%(q)s))      AS qvec,
    %(emb)s::vector(512)                        AS qemb
),
lex AS (
  SELECT c.chunk_id, c.doc_id, c.text, c.page_start, c.page_end, 
         c.content_type, c.image_path,
         ts_rank_cd(c.lex, to_tsquery('simple', regexp_replace(%(q_ts)s, '\s+', ' & ', 'g'))) AS lex_score,
         0::float AS vec_score
  FROM chunks c
  WHERE c.lex @@ to_tsquery('simple', regexp_replace(%(q_ts)s, '\s+', ' & ', 'g'))
  ORDER BY lex_score DESC
  LIMIT %(k_lex)s
),
vec AS (
  SELECT c.chunk_id, c.doc_id, c.text, c.page_start, c.page_end,
         c.content_type, c.image_path,
         0::float AS lex_score,
         1 - (c.emb <=> (SELECT qemb FROM q)) AS vec_score
  FROM chunks c
  ORDER BY c.emb <=> (SELECT qemb FROM q)
  LIMIT %(k_vec)s
),
u AS (
  SELECT * FROM lex
  UNION ALL
  SELECT * FROM vec
)
SELECT chunk_id, doc_id, text, page_start, page_end, content_type, image_path, lex_score, vec_score FROM u
ORDER BY (0.6*lex_score + 0.4*vec_score) DESC
LIMIT %(k)s;
"""

def sanitize_query_for_tsquery(query: str) -> str:
    """
    Sanitize query string for PostgreSQL tsquery to prevent syntax errors.
    
    Handles special characters that break tsquery syntax:
    - Replaces literal & with "and" (to avoid confusion with tsquery AND operator)
    - Removes/escapes other tsquery operators: |, !, (, ), :, *
    - Removes leading/trailing special characters
    - Strips bullet points and other formatting characters
    
    Args:
        query: Raw query string (potentially from LLM output)
        
    Returns:
        Sanitized query string safe for tsquery
    """
    import re
    
    # Remove leading bullet points, asterisks, dashes
    query = re.sub(r'^[\*\-\•\s]+', '', query.strip())
    
    # Replace literal & with "and" (preserve the meaning but avoid tsquery syntax conflicts)
    query = query.replace('&', ' and ')
    
    # Remove or escape other tsquery special characters
    # These characters have special meaning in tsquery: |, !, (, ), :, *
    # We'll remove them to avoid syntax errors, as they're not typically needed for basic search
    query = re.sub(r'[\!\|\:\*]', ' ', query)
    
    # Remove quotes that might cause issues
    query = query.replace('"', '').replace("'", '')
    
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    return query

def mmr(candidates, query_emb, lambda_mult=0.5, k=8):
    """Simple MMR over dense vectors only; you can blend in lex_score if desired."""
    selected, selected_ids = [], set()
    cand = candidates.copy()
    while len(selected) < min(k, len(cand)):
        best, best_id, best_score = None, None, -1e9
        for i, c in enumerate(cand):
            sim_q = float(np.dot(c["emb"], query_emb))  # cosine if normalized
            sim_d = max((np.dot(c["emb"], s["emb"]) for s in selected), default=0.0)
            score = lambda_mult*sim_q - (1-lambda_mult)*sim_d
            if score > best_score:
                best, best_id, best_score = c, c["chunk_id"], score
        selected.append(best)
        cand = [x for x in cand if x["chunk_id"] != best_id]
    return selected

def retrieve_hybrid(
    query: str, 
    k=8, 
    k_lex=40, 
    k_vec=40,
    query_image: Optional[Union[str, Image.Image]] = None
):
    """
    Hybrid retrieval with multi-modal support.
    
    Args:
        query: Text query string
        k: Number of results to return
        k_lex: Number of lexical results to retrieve
        k_vec: Number of vector results to retrieve
        query_image: Optional image to combine with text query for multi-modal search
        
    Returns:
        List of retrieved chunks with scores
    """
    # Embed query using CLIP (text or text+image)
    if query_image:
        qemb = embed_multi_modal(text=query, image_path=query_image, normalize_emb=True)
    else:
        qemb = embed_text(query, normalize_emb=True)
    
    # Convert numpy array to list of Python floats for psycopg2/pgvector compatibility
    # psycopg2 can't handle numpy.float32 directly - need Python native floats
    # For pgvector, we need to ensure the list is properly formatted
    if isinstance(qemb, np.ndarray):
        qemb_list = qemb.astype(np.float64).tolist()  # Convert to float64 then to list
    else:
        qemb_list = [float(x) for x in qemb]
    
    # Ensure we have exactly 512 dimensions
    if len(qemb_list) != 512:
        raise ValueError(f"Expected embedding dimension 512, got {len(qemb_list)}")
    
    # Sanitize query for tsquery to prevent syntax errors from special characters
    # This is especially important for LLM-generated refinement queries
    sanitized_query = sanitize_query_for_tsquery(query)
    
    # Log if sanitization changed the query (for debugging)
    if sanitized_query != query:
        logger.debug(f"Query sanitized: '{query}' -> '{sanitized_query}'")
    
    with connect() as conn, conn.cursor() as cur:
        try:
            cur.execute(HYBRID_SQL, {
                "q": query,  # Keep original for embedding/vector search
                "q_ts": sanitized_query,  # Use sanitized for tsquery
                "emb": qemb_list,
                "k": k_lex + k_vec,
                "k_lex": k_lex,
                "k_vec": k_vec
            })
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"SQL query failed: {e}", exc_info=True)
            conn.rollback()  # Explicitly rollback on error
            raise

    # Pull dense embeddings for MMR/rerank
    ids = [r[0] for r in rows]  # chunk_id
    if not ids: 
        return []
    
    def parse_vector(emb):
        """Parse pgvector vector type from database to numpy array."""
        if isinstance(emb, str):
            # pgvector returns vectors as strings like '[0.1,0.2,0.3]'
            # Remove brackets and split by comma
            try:
                # Remove brackets and whitespace
                emb_str = emb.strip('[]').strip()
                # Split by comma and convert to float
                # Handle scientific notation issues (e.g., "3.088634-05" -> "3.088634e-05")
                parts = [p.strip() for p in emb_str.split(',')]
                values = []
                for part in parts:
                    # Fix malformed scientific notation (missing 'e' before exponent)
                    # Pattern: number followed by - or + followed by digits (exponent)
                    # Match patterns like "3.088634-05" and convert to "3.088634e-05"
                    part = re.sub(r'([0-9])([+-])([0-9]+)$', r'\1e\2\3', part)
                    values.append(float(part))
                return np.array(values, dtype=np.float32)
            except Exception as e:
                logger.error(f"Failed to parse vector: {emb[:100]}... Error: {e}")
                raise ValueError(f"Could not parse vector from database: {e}")
        elif isinstance(emb, (list, tuple)):
            # Already a list/tuple, convert to numpy array
            return np.array(emb, dtype=np.float32)
        elif isinstance(emb, np.ndarray):
            # Already a numpy array
            return emb.astype(np.float32)
        else:
            raise ValueError(f"Unexpected vector type: {type(emb)}")
    
    with connect() as conn, conn.cursor() as cur:
        # Handle both old schema (no content_type) and new schema
        # Cast array to UUID[] to fix type mismatch error
        try:
            cur.execute(
                "SELECT chunk_id, text, emb, COALESCE(content_type, 'text'), COALESCE(image_path, '') FROM chunks WHERE chunk_id = ANY(%s::uuid[])", 
                (ids,)
            )
            id2 = {cid: (txt, parse_vector(emb), content_type or 'text', image_path or '') 
                   for cid, txt, emb, content_type, image_path in cur.fetchall()}
        except Exception:
            # Fallback for old schema
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
    if reranker and cands:
        pairs = [[query, c["text"]] for c in cands]
        try:
            ce_scores = reranker.predict(pairs)
            for c, s in zip(cands, ce_scores):
                c["ce"] = float(s)
            cands.sort(key=lambda x: x["ce"], reverse=True)
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Continuing without reranking.")

    # MMR diversify on top N, then return top-k
    diversified = mmr(cands[:20], qemb, lambda_mult=0.6, k=k)
    return diversified
