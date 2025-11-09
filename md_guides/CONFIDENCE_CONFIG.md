# Confidence Scoring Configuration

This document describes the environment variables used for calibrating confidence scoring in the Deep RAG system.

## Overview

The confidence scoring system uses a multi-feature approach to calculate confidence probabilities. The system extracts 10 features from retrieved chunks and uses a sigmoid function with configurable weights to determine confidence levels.

## Environment Variables

### Weight Configuration

Configure the weights for each feature using environment variables:

- `CONF_W0`: Bias weight (default: `-0.5`)
- `CONF_W1`: Weight for max rerank score (default: `2.2`)
- `CONF_W2`: Weight for margin between top scores (default: `1.0`)
- `CONF_W3`: Weight for mean cosine similarity (default: `1.6`)
- `CONF_W4`: Weight for cosine standard deviation (default: `-0.5`)
- `CONF_W5`: Weight for cosine coverage (default: `0.8`)
- `CONF_W6`: Weight for BM25 normalized (default: `1.2`)
- `CONF_W7`: Weight for term coverage (default: `0.8`)
- `CONF_W8`: Weight for unique page fraction (default: `0.6`)
- `CONF_W9`: Weight for document diversity (default: `0.6`)
- `CONF_W10`: Weight for answer overlap (default: `1.0`)

### Threshold Configuration

Configure decision thresholds:

- `CONF_ABSTAIN_TH`: Threshold below which to abstain (default: `0.45`)
  - If confidence < this threshold → Return "I don't know."
  
- `CONF_CLARIFY_TH`: Threshold for clarification (default: `0.65`)
  - If confidence < this threshold but >= ABSTAIN_TH → Return cautious answer with clarification
  - If confidence >= this threshold → Return full answer

## Features

The system calculates 10 features from retrieved chunks:

1. **f1 (max_rerank)**: Maximum reranking score (cross-encoder or vector)
2. **f2 (margin)**: Difference between top two rerank scores
3. **f3 (mean_cosine)**: Average cosine similarity across chunks
4. **f4 (cosine_std)**: Standard deviation of cosine similarity
5. **f5 (cos_coverage)**: Fraction of chunks with cosine >= 0.22
6. **f6 (bm25_norm)**: Normalized BM25/lexical scores
7. **f7 (term_coverage)**: Fraction of query terms found in chunks
8. **f8 (unique_page_frac)**: Fraction of unique pages
9. **f9 (doc_diversity)**: Fraction of unique documents
10. **f10 (answer_overlap)**: Jaccard similarity between answer and context (optional)

## Decision Logic

The confidence probability `p` is calculated as:

```
p = σ(w₀ + Σᵢ wᵢ × fᵢ)
```

Where `σ` is the sigmoid function.

Then the action is determined:

- `p < CONF_ABSTAIN_TH` → **abstain**: Return "I don't know."
- `CONF_ABSTAIN_TH ≤ p < CONF_CLARIFY_TH` → **clarify**: Return cautious answer
- `p ≥ CONF_CLARIFY_TH` → **answer**: Return full answer

## Example Configuration

Add to your `.env` file:

```bash
# Confidence weights
CONF_W0=-0.5
CONF_W1=2.2
CONF_W2=1.0
CONF_W3=1.6
CONF_W4=-0.5
CONF_W5=0.8
CONF_W6=1.2
CONF_W7=0.8
CONF_W8=0.6
CONF_W9=0.6
CONF_W10=1.0

# Decision thresholds
CONF_ABSTAIN_TH=0.45
CONF_CLARIFY_TH=0.65
```

## Calibration

For best results, calibrate these weights using labeled data:

1. Collect ~30-50 labeled questions (label: 0 = correct when answered, 1 = should abstain/clarify)
2. Run queries and log feature vectors
3. Fit logistic regression to learn optimal weights
4. Update environment variables with learned weights
5. Optionally grid-search thresholds to optimize F1 or minimize hallucination cost

The default weights provide a robust starting point and work well without calibration.

