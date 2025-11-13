# Confidence Heuristics  
### (Deep-RAG — Evidence-Based Guardrails for Safe LLM Responses)

This document explains how Deep-RAG computes and applies **confidence heuristics** to determine whether the LLM should answer a question, refine the query, or abstain.  
It is designed for internal documentation, engineering onboarding, and audits of retrieval‑based decision logic.

---

# 1. Purpose of the Confidence System

The goal of the confidence heuristic is to prevent:

- hallucinated answers  
- unsupported claims  
- contextless reasoning  
- misaligned responses during multi-step retrieval  

While allowing:

- safe fallback behaviors  
- iterative refinement (LangGraph loops)  
- dynamic thresholding during explicit document selection  
- evidence‑weighted context evaluation  

Confidence is always computed **before** the LLM is allowed to produce a final answer.

---

# 2. Inputs to the Confidence Calculation

Confidence is derived from the **final reranking stage**:

- Candidate chunks from hybrid retrieval  
- Cross‑encoder scores (`cross-encoder/ms-marco-MiniLM-L-6-v2`)  
- Optional blended score with hybrid values  
- Normalized final scores over Top‑K chunks  

Let:

- C_K(q) = {top‑K chunks after reranking}  
- s_final(q, cᵢ) = final score for each chunk (0–1)

Then confidence is a scalar in **[0, 100]**.

---

# 3. Core Formula

For any query q:

```
conf(q) = 100 * max_i s_final(q, cᵢ)
```

Where:

- s_final ∈ (0,1)
- i ranges over the final Top‑K retrieved chunks
- confidence is scaled to human‑interpretable range [0–100]

This formula gives a **single scalar** representing retrieval certainty.

---

# 4. Why Use the Maximum Score?

The highest‑scoring chunk is assumed to be:

1. The most relevant  
2. The most aligned with the query intent  
3. The most likely to provide evidence for an accurate answer  

Alternatives (e.g., mean, weighted average) were avoided because they:

- penalize multi-topic queries  
- collapse distributions  
- artificially lower confidence when many chunks are retrieved  

The max-score heuristic is stable and easy to audit.

---

# 5. Thresholds and Routing Behavior

Two thresholds govern system behavior:

### **Default threshold (no explicit doc selection):**
```
τ_default = 40
```

### **Explicit document selection threshold:**
```
τ_explicit = 30
```

### Routing rule:

```
If conf(q) ≥ τ:
    Proceed → synthesizer (LLM answer generation)
Else:
    Route → refine_retrieve (planner + improved retrieval)
```

---

# 6. Why Two Thresholds?

### Lower threshold during explicit selection:
When the user explicitly identifies files, pages, or attachments:

- the system trusts retrieval more  
- hallucination risk decreases  
- chunk space is scoped by user intent  

Thus, we allow answering even when confidence is slightly weaker.

### Higher threshold during free-search:
When the system selects documents freely:

- hallucination risk increases  
- retrieval may include unrelated documents  
- larger chunk spaces reduce precision  

Thus, we require stronger evidence before answering.

---

# 7. Decision Boundary Visualization

```
Confidence (0–100)
 ─────────────────────────────────────►

 0–29        30–39            40–100
 Low         Gray             High
 └────────┬─────────┬───────────────┘
          │         │
          │         └── Default OK only if explicit
          │
          └── Always refine & re-retrieve
```

Interpretation:

- **0–29** → Never answer  
- **30–39** → Answer only if explicit-selection mode  
- **40+** → Safe to answer normally  

---

# 8. Integration with LangGraph

Within the LangGraph pipeline:

```
retriever → compressor → critic → (refine_retrieve or synthesizer)
```

The critic node evaluates:

- s_final from reranker  
- conf(q) based on max-score  
- threshold τ based on whether the user selected specific documents  

Routing logic (pseudocode):

```
if conf(q) >= τ:
    route = "synthesize"
else:
    route = "refine_retrieve"
```

The refine loop continues until:

- confidence stabilizes, or  
- max iteration depth is reached  

---

# 9. Edge Cases & Guards

### 9.1 Multiple top chunks share identical scores
Behavior: still safe — only the **maximum** matters.

### 9.2 No valid chunks retrieved
System sets:

```
conf(q) = 0
```

Then routes to refinement.

### 9.3 Cross-encoder abstains or output degenerate
System falls back to hybrid score:

```
s_final = s_hyb
```

Confidence remains constrained to [0–100].

### 9.4 All chunks score extremely low (e.g., <0.1)
Confidence stays below threshold → forces refinement.

---

# 10. Rationale Behind the Heuristic

The heuristic is chosen because:

- It is **robust** against noisy retrieval  
- It captures **semantic alignment** (via CLIP + reranker)  
- It is **interpretable** (simple score scaling)  
- It is **cheap** (no complex probability aggregation)  
- It works well in multimodal contexts (text + images + PDFs)  

Most importantly:

> It guarantees the LLM never answers without at least one high-quality evidence chunk.

---

# 11. Summary

✓ Confidence = 100 × max(final reranked score)  
✓ Final reranked score comes from a cross-encoder  
✓ Thresholds = 40 (default) and 30 (explicit-document mode)  
✓ Routing: refine if below threshold, synthesize if above  
✓ Prevents hallucinations and enforces evidence-based answers  
✓ Fully integrated into LangGraph’s critic → refine loop  

---

# End of File
