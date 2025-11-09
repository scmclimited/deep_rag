"""
Maximal Marginal Relevance (MMR) for diversity in retrieval.
"""
import numpy as np
from typing import List, Dict


def mmr(candidates: List[Dict], query_emb: np.ndarray, lambda_mult: float = 0.5, k: int = 8) -> List[Dict]:
    """
    Simple MMR over dense vectors only; you can blend in lex_score if desired.
    
    Args:
        candidates: List of candidate chunks with 'emb' and 'chunk_id' keys
        query_emb: Query embedding vector
        lambda_mult: Lambda parameter for MMR (0.0 = pure diversity, 1.0 = pure relevance)
        k: Number of results to return
        
    Returns:
        List of selected chunks with diversity
    """
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

