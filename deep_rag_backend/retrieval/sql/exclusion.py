"""
Hybrid SQL query generation with doc_id exclusion.
"""
from typing import Optional


def get_hybrid_sql_with_exclusion(embedding_dim: int, exclude_doc_id: Optional[str] = None) -> str:
    """
    Generate hybrid SQL query with optional doc_id exclusion.
    
    Args:
        embedding_dim: Embedding dimension (768 for openai/clip-vit-large-patch14-336, 512 for CLIP-ViT-B/32)
        exclude_doc_id: Optional document ID to exclude from results
        
    Returns:
        SQL query string with optional doc_id exclusion filter
    """
    # Add exclusion filter if provided
    exclude_filter = "AND c.doc_id != %(exclude_doc_id)s" if exclude_doc_id else ""
    
    return f"""
WITH
q AS (
  SELECT
    to_tsvector('simple', unaccent(%(q)s))      AS qvec,
    %(emb)s::vector({embedding_dim})             AS qemb
),
lex AS (
  SELECT c.chunk_id, c.doc_id, c.text, c.page_start, c.page_end, 
         c.content_type, c.image_path,
         ts_rank_cd(c.lex, to_tsquery('simple', regexp_replace(%(q_ts)s, '\\s+', ' & ', 'g'))) AS lex_score,
         0::float AS vec_score
  FROM chunks c
  WHERE c.lex @@ to_tsquery('simple', regexp_replace(%(q_ts)s, '\\s+', ' & ', 'g'))
    {exclude_filter}
  ORDER BY lex_score DESC
  LIMIT %(k_lex)s
),
vec AS (
  SELECT c.chunk_id, c.doc_id, c.text, c.page_start, c.page_end,
         c.content_type, c.image_path,
         0::float AS lex_score,
         1 - (c.emb <=> (SELECT qemb FROM q)) AS vec_score
  FROM chunks c
  WHERE 1=1 {exclude_filter}
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

