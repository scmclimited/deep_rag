"""
Confidence scoring module for RAG systems.
Implements multi-feature confidence calculation with environment variable calibration.
"""
import os
import math
import logging
from typing import List, Dict, Any, Optional, Set
from dotenv import load_dotenv

load_dotenv()

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
    "w0": float(os.getenv("CONF_W0", "-.08")),   # Lower bias to allow more positive scores
    "w1": float(os.getenv("CONF_W1", "3.0")),    # max_rerank (increased - strong indicator)
    "w2": float(os.getenv("CONF_W2", "1.5")),    # margin (increased - good separation)
    "w3": float(os.getenv("CONF_W3", "2.2")),    # mean_cosine (increased - important)
    "w4": float(os.getenv("CONF_W4", "-0.3")),   # cosine_std (less negative - variance can be okay)
    "w5": float(os.getenv("CONF_W5", "1.0")),    # cos_coverage (increased - more important)
    "w6": float(os.getenv("CONF_W6", "1.5")),    # bm25_norm (increased - lexical match important)
    "w7": float(os.getenv("CONF_W7", "1.4")),    # term_coverage (increased - query terms found)
    "w8": float(os.getenv("CONF_W8", "0.8")),    # unique_page_frac (increased - more important)
    "w9": float(os.getenv("CONF_W9", "0.4")),    # doc_diversity (increased - less important)
    "w10": float(os.getenv("CONF_W10", "1.4")),  # answer_overlap (increased - good indicator)
}

# Decision thresholds (adjusted to be less strict)
# Lower thresholds allow correct answers to pass through
# override via environment variables
ABSTAIN_TH = float(os.getenv("CONF_ABSTAIN_TH", "0.20"))  # Lowered from 0.45 to 0.30 (30%)
CLARIFY_TH = float(os.getenv("CONF_CLARIFY_TH", "0.60"))  # Lowered from 0.65 to 0.55 (55%)

# Log loaded weights and thresholds
logger.info(f"Confidence weights loaded: w0={_W['w0']}, w1={_W['w1']}, w2={_W['w2']}, w3={_W['w3']}, w4={_W['w4']}, "
            f"w5={_W['w5']}, w6={_W['w6']}, w7={_W['w7']}, w8={_W['w8']}, w9={_W['w9']}, w10={_W['w10']}")
logger.info(f"Confidence thresholds: ABSTAIN_TH={ABSTAIN_TH}, CLARIFY_TH={CLARIFY_TH}")


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
        logger.debug("build_conf_features: No chunks provided, returning zero features")
        return {f"f{i}": 0.0 for i in range(1, 11)}
    
    logger.debug(f"build_conf_features: Building features from {k} chunks")
    
    # Extract scores (using ce as rerank_score, vec as cosine)
    reranks = [float(c.get("ce", c.get("vec", 0.0)) or 0.0) for c in ranked_chunks]
    cosines = [float(c.get("vec", 0.0) or 0.0) for c in ranked_chunks]
    lex_scores = [float(c.get("lex", 0.0) or 0.0) for c in ranked_chunks]
    # Extract actual CE scores (not fallback to vec) to detect if all are negative
    ce_scores = [float(c.get("ce", 0.0) or 0.0) for c in ranked_chunks]
    
    logger.debug(f"build_conf_features: Score ranges - reranks: [{min(reranks) if reranks else 0:.3f}, {max(reranks) if reranks else 0:.3f}], "
                 f"cosines: [{min(cosines) if cosines else 0:.3f}, {max(cosines) if cosines else 0:.3f}], "
                 f"lex: [{min(lex_scores) if lex_scores else 0:.3f}, {max(lex_scores) if lex_scores else 0:.3f}], "
                 f"ce: [{min(ce_scores) if ce_scores else 0:.3f}, {max(ce_scores) if ce_scores else 0:.3f}]")
    
    # Check if lexical search failed but vector search found relevant chunks
    # This happens with meta-queries like "find documents with X" where lexical requires all terms
    # but vector search successfully finds chunks containing X
    # Also happens with explicitly selected documents and ambiguous queries like "share details about this document"
    has_lexical_matches = any(lex > 0.0 for lex in lex_scores)
    # Lower threshold (0.4) to catch moderate vector matches that are still relevant
    # Especially important for explicitly selected documents with ambiguous queries
    has_good_vector_matches = any(vec > 0.4 for vec in cosines)  # Lowered from 0.5 to catch more cases
    # Check if all CE scores are negative (indicating meta-query mismatch)
    all_ce_negative = len(ce_scores) > 0 and all(ce < 0.0 for ce in ce_scores)
    
    # If lexical failed but vector found good matches, and all CE are negative,
    # this is likely a meta-query issue or explicit doc selection with ambiguous query
    # Use vector scores instead of CE for rerank
    if not has_lexical_matches and has_good_vector_matches and all_ce_negative:
        logger.info(f"Lexical search failed but vector search found relevant chunks - likely meta-query or explicit doc selection. "
                   f"Using vector scores for rerank instead of CE. "
                   f"has_lexical_matches={has_lexical_matches}, has_good_vector_matches={has_good_vector_matches}, "
                   f"all_ce_negative={all_ce_negative}, max_vec={max(cosines) if cosines else 0:.3f}")
        # Use vector scores as rerank scores when CE is unreliable (meta-query/explicit selection scenario)
        reranks = cosines.copy()
        logger.info(f"Replaced rerank scores: max_rerank changed from {max([float(c.get('ce', c.get('vec', 0.0)) or 0.0) for c in ranked_chunks]) if ranked_chunks else 0:.3f} to {max(reranks) if reranks else 0:.3f}")
    else:
        logger.debug(f"Not using vector scores for rerank: has_lexical_matches={has_lexical_matches}, "
                    f"has_good_vector_matches={has_good_vector_matches}, all_ce_negative={all_ce_negative}")
    
    # f1: max rerank score (raw value, weights applied in confidence_probability)
    max_r = max(reranks) if reranks else 0.0
    
    # f2: margin (difference between top two rerank scores)
    # If only one chunk, margin is 0.0 (no separation)
    if k > 1:
        sec_r = sorted(reranks, reverse=True)[1]
        margin = max_r - sec_r
    else:
        margin = 0.0
    
    # f3: mean cosine similarity (raw value)
    mean_cos = sum(cosines) / k if k > 0 else 0.0
    
    # f4: standard deviation of cosine similarity (raw value)
    if k > 1:
        var_cos = sum((x - mean_cos) ** 2 for x in cosines) / k
        std_cos = math.sqrt(var_cos)
    else:
        # For single chunk, std is 0 (no variance)
        std_cos = 0.0
    
    # f5: cosine coverage (fraction over a small floor) - raw value
    COS_FLOOR = 0.22
    cos_cov = sum(1 for x in cosines if x >= COS_FLOOR) / k if k > 0 else 0.0
    
    # f6: BM25 normalized (if available, otherwise 0.0) - raw value
    # Note: We can approximate with lex scores normalized
    # (lex_scores already extracted above)
    if lex_scores and k > 0:
        max_lex = max(lex_scores)
        bm25_norm = sum(lex_scores) / (max_lex * k) if max_lex > 0 else 0.0
    else:
        bm25_norm = 0.0
    
    # f7: term coverage (query terms found in chunks) - raw value
    # For meta-queries like "find documents with X", focus on finding the actual search term (X)
    # rather than requiring all query words
    if query_terms:
        seen_terms = set()
        # Filter out common stop words that don't indicate relevance
        # These are often in meta-queries but not in content
        stop_words = {"can", "you", "find", "me", "which", "documents", "have", "in", "them", "the", "a", "an", "is", "are", "was", "were", "do", "does", "did"}
        meaningful_terms = {t.lower() for t in query_terms if t.lower() not in stop_words}
        
        # If we filtered out too many terms, use original terms (query might be short)
        if len(meaningful_terms) == 0:
            meaningful_terms = {t.lower() for t in query_terms}
        
        for c in ranked_chunks:
            text = (c.get("text") or "").lower()
            # Simple tokenization
            tokens = set(text.split())
            seen_terms |= (tokens & meaningful_terms)
        term_cov = _safe_div(len(seen_terms), len(meaningful_terms)) if meaningful_terms else 0.0
    else:
        term_cov = 0.0
    
    # f8: unique page fraction (count unique page numbers, not page ranges) - raw value
    # Count unique p0 values (starting page numbers) to match test expectations
    unique_page_numbers = len(set(c.get("p0") for c in ranked_chunks if c.get("p0") is not None))
    page_frac = _safe_div(unique_page_numbers, k)
    
    # f9: document diversity - raw value
    # This represents how concentrated the chunks are in terms of documents
    # Use unique_docs/k for consistency with test expectations
    # Note: For single document, this gives 1/k (e.g., 1/3 = 0.333)
    # For multiple documents, this gives unique_docs/k (diversity ratio)
    unique_docs = len(set(c.get("doc_id") for c in ranked_chunks if c.get("doc_id")))
    doc_div = _safe_div(unique_docs, k)
    
    # f10: answer overlap (optional, computed after draft answer) - raw value
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
    
    feats = {
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
    
    logger.debug(f"build_conf_features: Raw features computed - f1={max_r:.3f}, f2={margin:.3f}, f3={mean_cos:.3f}, "
                 f"f4={std_cos:.3f}, f5={cos_cov:.3f}, f6={bm25_norm:.3f}, f7={term_cov:.3f}, "
                 f"f8={page_frac:.3f}, f9={doc_div:.3f}, f10={overlap:.3f}")
    
    return feats


def confidence_probability(feats: Dict[str, float]) -> float:
    """
    Calculate confidence probability from features using sigmoid.
    
    Args:
        feats: Dictionary of features f1-f10
        
    Returns:
        Confidence probability between 0 and 1
    """
    s = _W["w0"]  # bias
    weighted_contributions = {}
    for i in range(1, 11):
        weight = _W.get(f"w{i}", 0.0)
        feat_value = feats.get(f"f{i}", 0.0)
        contribution = weight * feat_value
        weighted_contributions[f"w{i}*f{i}"] = contribution
        s += contribution
    
    prob = _sigmoid(s)
    logger.debug(f"confidence_probability: Weighted sum={s:.3f}, probability={prob:.3f}")
    logger.debug(f"confidence_probability: Top contributions - "
                 f"{', '.join([f'{k}={v:.3f}' for k, v in sorted(weighted_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]])}")
    
    return prob


def decide_action(p: float) -> str:
    """
    Decide action based on confidence probability.
    
    Args:
        p: Confidence probability (0-1)
        
    Returns:
        Action: "abstain", "clarify", or "answer"
    """
    if p < ABSTAIN_TH:
        logger.debug(f"decide_action: p={p:.3f} < ABSTAIN_TH={ABSTAIN_TH} → abstain")
        return "abstain"
    if p < CLARIFY_TH:
        logger.debug(f"decide_action: p={p:.3f} < CLARIFY_TH={CLARIFY_TH} → clarify")
        return "clarify"
    logger.debug(f"decide_action: p={p:.3f} >= CLARIFY_TH={CLARIFY_TH} → answer")
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
    
    logger.info(f"get_confidence_for_chunks: {len(ranked_chunks)} chunks → confidence={confidence_percentage:.1f}%, "
                f"probability={p:.3f}, action={action}")
    
    return {
        "confidence": round(confidence_percentage, 2),  # Percentage for display
        "probability": p,  # Probability (0-1) for internal use
        "action": action,
        "features": feats,
        "abstain_threshold": ABSTAIN_TH,
        "clarify_threshold": CLARIFY_TH
    }

