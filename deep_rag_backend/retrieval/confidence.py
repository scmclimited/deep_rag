"""
Confidence scoring module for RAG systems.
Implements multi-feature confidence calculation with environment variable calibration.
"""
import os
import math
import logging
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)


def _safe_div(a: float, b: float) -> float:
    """Safe division that returns 0.0 if denominator is 0."""
    return a / b if b else 0.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    try:
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        z = math.exp(x)
        return z / (1 + z)
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# Default weights (calibrated for better accuracy); override via env or learned calibration
# Adjusted to be less conservative - correct answers should get higher confidence
_W = {
    "w0": float(os.getenv("CONF_W0", "-1.0")),  # Lower bias to allow more positive scores
    "w1": float(os.getenv("CONF_W1", "3.0")),  # max_rerank (increased - strong indicator)
    "w2": float(os.getenv("CONF_W2", "1.5")),  # margin (increased - good separation)
    "w3": float(os.getenv("CONF_W3", "2.0")),  # mean_cosine (increased - important)
    "w4": float(os.getenv("CONF_W4", "-0.3")),  # cosine_std (less negative - variance can be okay)
    "w5": float(os.getenv("CONF_W5", "1.0")),  # cos_coverage (increased)
    "w6": float(os.getenv("CONF_W6", "1.5")),  # bm25_norm (increased - lexical match important)
    "w7": float(os.getenv("CONF_W7", "1.2")),  # term_coverage (increased - query terms found)
    "w8": float(os.getenv("CONF_W8", "0.8")),  # unique_page_frac (increased)
    "w9": float(os.getenv("CONF_W9", "0.4")),  # doc_diversity (decreased - less important)
    "w10": float(os.getenv("CONF_W10", "1.2")),  # answer_overlap (increased - good indicator)
}

# Decision thresholds (adjusted to be less strict)
# Lower thresholds allow correct answers to pass through
ABSTAIN_TH = float(os.getenv("CONF_ABSTAIN_TH", "0.30"))  # Lowered from 0.45 to 0.30 (30%)
CLARIFY_TH = float(os.getenv("CONF_CLARIFY_TH", "0.55"))  # Lowered from 0.65 to 0.55 (55%)


def build_conf_features(
    ranked_chunks: List[Dict[str, Any]],
    query_terms: Optional[Set[str]] = None,
    answer_text: Optional[str] = None,
    use_answer_overlap: bool = False,
) -> Dict[str, float]:
    """
    Build confidence features from ranked chunks.
    
    Args:
        ranked_chunks: List of chunks with scores (each: {ce, vec, lex, doc_id, p0, p1, text, ...})
        query_terms: Optional set of query terms for lexical features
        answer_text: Optional answer text for overlap feature (f10)
        use_answer_overlap: Whether to compute answer overlap feature
        
    Returns:
        Dictionary of features f1-f10
    """
    k = len(ranked_chunks)
    if k == 0:
        return {f"f{i}": 0.0 for i in range(1, 11)}
    
    # Extract scores (using ce as rerank_score, vec as cosine)
    reranks = [float(c.get("ce", c.get("vec", 0.0)) or 0.0) for c in ranked_chunks]
    cosines = [float(c.get("vec", 0.0) or 0.0) for c in ranked_chunks]
    
    # f1: max rerank score
    max_r = max(reranks) if reranks else 0.0
    
    # f2: margin (difference between top two rerank scores)
    # If only one chunk, margin is 0.0 (no separation)
    if k > 1:
        sec_r = sorted(reranks, reverse=True)[1]
        margin = max_r - sec_r
    else:
        margin = 0.0
    
    # f3: mean cosine similarity
    mean_cos = sum(cosines) / k if k > 0 else 0.0
    
    # f4: standard deviation of cosine similarity
    if k > 1:
        var_cos = sum((x - mean_cos) ** 2 for x in cosines) / k
        std_cos = math.sqrt(var_cos)
    else:
        std_cos = 0.0
    
    # f5: cosine coverage (fraction over a small floor)
    COS_FLOOR = 0.22
    cos_cov = sum(1 for x in cosines if x >= COS_FLOOR) / k if k > 0 else 0.0
    
    # f6: BM25 normalized (if available, otherwise 0.0)
    # Note: We can approximate with lex scores normalized
    lex_scores = [float(c.get("lex", 0.0) or 0.0) for c in ranked_chunks]
    if lex_scores:
        max_lex = max(lex_scores)
        bm25_norm = sum(lex_scores) / (max_lex * k) if max_lex > 0 else 0.0
    else:
        bm25_norm = 0.0
    
    # f7: term coverage (query terms found in chunks)
    if query_terms:
        seen_terms = set()
        for c in ranked_chunks:
            text = (c.get("text") or "").lower()
            # Simple tokenization
            tokens = set(text.split())
            seen_terms |= (tokens & set(t.lower() for t in query_terms))
        term_cov = _safe_div(len(seen_terms), len(query_terms))
    else:
        term_cov = 0.0
    
    # f8: unique page fraction (count unique page numbers, not page ranges)
    # Count unique p0 values (starting page numbers) to match test expectations
    unique_page_numbers = len(set(c.get("p0") for c in ranked_chunks if c.get("p0") is not None))
    page_frac = _safe_div(unique_page_numbers, k)
    
    # f9: document diversity
    # This represents how concentrated the chunks are in terms of documents
    # Use unique_docs/k for consistency with test expectations
    # Note: For single document, this gives 1/k (e.g., 1/3 = 0.333)
    # For multiple documents, this gives unique_docs/k (diversity ratio)
    unique_docs = len(set(c.get("doc_id") for c in ranked_chunks if c.get("doc_id")))
    doc_div = _safe_div(unique_docs, k)
    
    # f10: answer overlap (optional, computed after draft answer)
    if use_answer_overlap and answer_text:
        ans_tokens = set(answer_text.lower().split())
        ctx_tokens = set()
        for c in ranked_chunks:
            text = (c.get("text") or "").lower()
            ctx_tokens |= set(text.split())
        # Jaccard similarity
        inter = len(ans_tokens & ctx_tokens)
        union = len(ans_tokens | ctx_tokens) or 1
        overlap = inter / union
    else:
        overlap = 0.0
    
    return {
        "f1": max_r,
        "f2": margin,
        "f3": mean_cos,
        "f4": std_cos,
        "f5": cos_cov,
        "f6": bm25_norm,
        "f7": term_cov,
        "f8": page_frac,
        "f9": doc_div,
        "f10": overlap,
    }


def confidence_probability(feats: Dict[str, float]) -> float:
    """
    Calculate confidence probability from features using sigmoid.
    
    Args:
        feats: Dictionary of features f1-f10
        
    Returns:
        Confidence probability between 0 and 1
    """
    s = _W["w0"]  # bias
    for i in range(1, 11):
        s += _W.get(f"w{i}", 0.0) * feats.get(f"f{i}", 0.0)
    return _sigmoid(s)


def decide_action(p: float) -> str:
    """
    Decide action based on confidence probability.
    
    Args:
        p: Confidence probability (0-1)
        
    Returns:
        Action: "abstain", "clarify", or "answer"
    """
    if p < ABSTAIN_TH:
        return "abstain"
    if p < CLARIFY_TH:
        return "clarify"
    return "answer"


def get_confidence_for_chunks(
    ranked_chunks: List[Dict[str, Any]],
    query: Optional[str] = None,
    answer_text: Optional[str] = None,
    use_answer_overlap: bool = False,
) -> Dict[str, Any]:
    """
    Get confidence score and decision for ranked chunks.
    
    Args:
        ranked_chunks: List of chunks with scores
        query: Optional query text for term extraction
        answer_text: Optional answer text for overlap feature
        use_answer_overlap: Whether to compute answer overlap
        
    Returns:
        Dictionary with confidence, probability, action, and features
    """
    # Extract query terms if query provided
    query_terms = None
    if query:
        query_terms = set(query.lower().split())
    
    # Build features
    feats = build_conf_features(
        ranked_chunks,
        query_terms=query_terms,
        answer_text=answer_text,
        use_answer_overlap=use_answer_overlap
    )
    
    # Calculate probability
    p = confidence_probability(feats)
    
    # Decide action
    action = decide_action(p)
    
    # Return confidence as percentage for display, but keep probability for internal use
    confidence_percentage = p * 100
    
    return {
        "confidence": round(confidence_percentage, 2),  # Percentage for display
        "probability": p,  # Probability (0-1) for internal use
        "action": action,
        "features": feats,
        "abstain_threshold": ABSTAIN_TH,
        "clarify_threshold": CLARIFY_TH
    }

