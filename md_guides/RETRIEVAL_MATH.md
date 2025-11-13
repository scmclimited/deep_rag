# Retrieval, Weighting & Reranking Math  
### (Deep-RAG — Semantic + Lexical + Cross-Encoder + LangGraph RAG)

This document describes the mathematical foundations behind Deep-RAG’s hybrid retrieval pipeline, weighting logic, cross-encoder reranking, and the LangGraph-based RAG inference loop.

It covers:

- CLIP embedding similarity  
- Lexical BM25-style scoring  
- Hybrid weighting  
- Cross-encoder reranking  
- Confidence heuristics  
- LangGraph agentic RAG equations  
- Descriptions for **K**, **vec**, **lex**, and reranker scores  

Everything here directly reflects the retrieval logic implemented in the backend.

---

# 0. Notation

| Symbol | Meaning |
|-------|---------|
| q | User query (text or text+image) |
| c_i | Candidate chunk |
| e_q, e_{c_i} | Embeddings from `openai/clip-vit-large-patch14-336` |
| s_vec | Vector similarity score |
| s_lex | Lexical score (BM25/pg_trgm) |
| s_hyb | Hybrid score (vec + lex) |
| s_ce | Cross-encoder score from `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| s_final | Final ranking score |
| K_LEX, K_VEC, K_RETRIEVER | Retrieval-width parameters |

---

# 1. Embedding Model  
### `openai/clip-vit-large-patch14-336` (Multi-modal, 768-dim)

CLIP embeds both text **and** images into a shared 768-dimensional space.

### 1.1 Query embedding
e_q = f_CLIP_text(q)

### 1.2 Chunk embedding
- Text chunk:  
  e_{c_i} = f_CLIP_text(c_i)

- Image or figure:  
  e_{c_i} = f_CLIP_image(image_i)

### Descriptor
CLIP creates a unified embedding space for documents, screenshots, figures, tables, and text—allowing multi-modal retrieval.

---

# 2. Vector Similarity  
### Cosine similarity over CLIP embeddings

s_vec(q, c_i) = (e_q · e_{c_i}) / (||e_q|| * ||e_{c_i}||)

Normalize:

tilde_s_vec = (s_vec + 1) / 2

### Descriptor
Measures semantic closeness independent of exact wording.

---

# 3. Lexical Score  
### BM25-style formulation

s_lex(q, c_i) = Σ_t IDF(t) * [ f(t,c_i)*(k1+1) / ( f(t,c_i) + k1*(1 - b + b*|c_i|/avgdl ) ) ]

Normalize:

tilde_s_lex = (s_lex - min) / (max - min + eps)

### Descriptor
Captures keyword, phrase, and exact-match relevance.

---

# 4. K Parameters

- K_LEX — top lexical documents  
- K_VEC — top vector documents  
- K_RETRIEVER — final set passed to reranker

---

# 5. Hybrid Retrieval  

C_LEX = Top-K_LEX(s_lex)  
C_VEC = Top-K_VEC(s_vec)  
C = C_LEX ∪ C_VEC

Hybrid score:

s_hyb = λ * tilde_s_vec + (1 - λ) * tilde_s_lex

Final retrieved set:

C_K = Top-K_RETRIEVER(s_hyb)

---

# 6. Cross-Encoder Reranking  
### `cross-encoder/ms-marco-MiniLM-L-6-v2`

z_i = f_CE([q; c_i])  
s_ce = sigmoid(z_i)

Final score options:

- Pure: s_final = s_ce  
- Blended: s_final = α*s_ce + (1-α)*s_hyb

---

# 7. Confidence Score

conf(q) = 100 * max_i s_final(q, c_i)

LLM is invoked if:

conf(q) ≥ τ

(Default τ = 40, explicit doc selection τ = 30)

---

# 8. LangGraph Agentic RAG  

Iterative refinement:

q⁰ = q

For t in 0…T:
  Cᵗ = HybridRetrieve(qᵗ)
  ctxᵗ = Compress(Cᵗ)
  confᵗ = conf(qᵗ)

  If confᵗ ≥ τ: break

  qᵗ⁺¹ = RefineQuery(qᵗ, ctxᵗ)

Answer:

y = LLM(qᵗ, ctxᵗ)

---

# 9. Score Descriptions

- vec score — CLIP semantic similarity  
- lex score — BM25-style keyword density  
- hybrid — weighted vec + lex  
- reranker — cross-encoder probability  
- final score — used to order context for LLM  

---

# 10. ASCII Pipeline

Query  
 → planner  
 → retriever  
 → hybrid vec+lex  
 → cross-encoder  
 → compressor  
 → critic  
   → (loop refine)  
 → synthesizer  
 → citation-pruner  
 → answer

---

# End of File
