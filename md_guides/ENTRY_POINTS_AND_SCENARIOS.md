# Entry Points and Scenarios

This guide explains when and how to use each entry point in Deep RAG, including detailed scenarios and use cases.

---

## Overview

Deep RAG provides multiple entry points (CLI, Make, TOML, REST API) for different workflows. Each entry point supports different combinations of:
- **Ingestion**: Adding documents to the knowledge base
- **Querying**: Asking questions over the knowledge base
- **Pipeline Type**: Direct (linear) or LangGraph (conditional routing)
- **Document Filtering**: `doc_id` for document-specific retrieval
- **Cross-Document Retrieval**: `--cross-doc` flag for two-stage retrieval

---

## Entry Point Comparison

| Entry Point | Ingestion | Query | Pipeline | `doc_id` Support | `--cross-doc` Support | `--thread-id` Support |
|------------|-----------|-------|----------|-----------------|----------------------|----------------------|
| `ingest` / `POST /ingest` | ✅ | ❌ | N/A | Generated | ❌ | ❌ |
| `query` / `POST /ask` | ❌ | ✅ | Direct | Optional | ✅ | ❌ |
| `query-graph` / `POST /ask-graph` | ❌ | ✅ | LangGraph | Optional | ✅ | ✅ |
| `infer` / `POST /infer` | ✅ | ✅ | Direct | Generated on-the-fly | ✅ | ❌ |
| `infer-graph` / `POST /infer-graph` | ✅ | ✅ | LangGraph | Generated on-the-fly | ✅ | ✅ |

---

## Scenarios

### Scenario 1: Pre-Populate Knowledge Base

**Use Case**: You want to ingest documents into the knowledge base without querying them immediately.

**Entry Points**:
- `ingest` / `make cli-ingest` / `POST /ingest`

**Example**:
```bash
# CLI (from deep_rag_backend directory)
cd deep_rag_backend
python -m inference.cli ingest "path/to/file.pdf"
python -m inference.cli ingest "path/to/file.pdf" --title "Custom Title"

# Make (from project root)
make cli-ingest FILE="path/to/file.pdf" DOCKER=true
make cli-ingest FILE="path/to/file.pdf" DOCKER=true TITLE="Custom Title"

# REST API (from any directory)
curl -X POST http://localhost:8000/ingest \
  -F "attachment=@path/to/file.pdf" \
  -F "title=Optional Document Title"
```

**What Happens**:
1. Document is ingested → `doc_id` is generated
2. Chunks are created and embedded
3. Chunks are inserted into the database with `doc_id` reference
4. **`doc_id` is returned** in the response

**When to Use**:
- Batch ingestion workflows
- Pre-populating knowledge base before queries
- Storing `doc_id` for later queries (Scenario 3)

---

### Scenario 2: Query Existing Documents (Simple Questions)

**Use Case**: You want to ask simple, straightforward questions over existing documents.

**Entry Points**:
- `query` / `make query` / `POST /ask`

**Example**:
```bash
# CLI (from deep_rag_backend directory)
cd deep_rag_backend
# Query all documents
python -m inference.cli query "What are the main sections?"

# Query specific document
python -m inference.cli query "What are the requirements?" --doc-id 550e8400-e29b-41d4-a716-446655440000

# Query with cross-doc enabled
python -m inference.cli query "What are the requirements?" --doc-id 550e8400-e29b-41d4-a716-446655440000 --cross-doc

# Make (from project root)
make query Q="What are the requirements?" DOCKER=true
make query Q="What are the requirements?" DOCKER=true DOC_ID=550e8400-e29b-41d4-a716-446655440000
make query Q="What are the requirements?" DOCKER=true DOC_ID=550e8400-e29b-41d4-a716-446655440000 CROSS_DOC=true

# REST API (from any directory)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the requirements?", "doc_id": "550e8400-e29b-41d4-a716-446655440000", "cross_doc": true}'
```

**What Happens**:
1. Query is executed using the direct pipeline (linear execution)
2. If `doc_id` is provided: Retrieval is filtered to that specific document
3. If `--cross-doc` is enabled: Two-stage retrieval (see `--cross-doc` flag documentation)
4. Answer is generated with document context when available

**When to Use**:
- Simple, straightforward questions
- Fast, deterministic answers
- No need for iterative refinement

---

### Scenario 3: Query Existing Documents (Complex Questions)

**Use Case**: You want to ask complex questions that may require iterative refinement and multiple retrieval passes.

**Entry Points**:
- `query-graph` / `make query-graph` / `POST /ask-graph`

**Example**:
```bash
# CLI (from deep_rag_backend directory)
cd deep_rag_backend
# Query all documents
python -m inference.cli query-graph "What are the specific requirements?" --thread-id session-1

# Query specific document
python -m inference.cli query-graph "What are the specific requirements?" --doc-id 550e8400-e29b-41d4-a716-446655440000 --thread-id session-1

# Query with cross-doc enabled
python -m inference.cli query-graph "What are the specific requirements?" --doc-id 550e8400-e29b-41d4-a716-446655440000 --thread-id session-1 --cross-doc

# Make (from project root)
make query-graph Q="What are the specific requirements?" DOCKER=true THREAD_ID=session-1
make query-graph Q="What are the specific requirements?" DOCKER=true THREAD_ID=session-1 DOC_ID=550e8400-e29b-41d4-a716-446655440000 CROSS_DOC=true

# REST API (from any directory)
curl -X POST http://localhost:8000/ask-graph \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the specific requirements?", "thread_id": "session-1", "doc_id": "550e8400-e29b-41d4-a716-446655440000", "cross_doc": true}'
```

**What Happens**:
1. Query is executed using the LangGraph pipeline (conditional routing)
2. Agents can refine queries and retrieve more evidence if confidence is low
3. If `doc_id` is provided: Retrieval is filtered to that specific document
4. If `--cross-doc` is enabled: Two-stage retrieval (see `--cross-doc` flag documentation)
5. Answer is generated with document context when available
6. **Reasoning logs are saved** to `deep_rag_backend/inference/graph/logs/` for SFT training (production) or `deep_rag_backend/inference/graph/logs/test_logs/` for test executions

**When to Use**:
- Complex questions requiring multi-step reasoning
- Questions that may need iterative refinement
- When you want reasoning logs for future training
- When you need conversation state (`--thread-id`)

---

### Scenario 4: Ingest + Query (Simple Questions)

**Use Case**: You have a document and want to ingest it and immediately ask simple questions about it.

**Entry Points**:
- `infer` / `make infer` / `POST /infer`

**Example**:
```bash
# CLI (from deep_rag_backend directory)
cd deep_rag_backend
# Ingest + Query
python -m inference.cli infer "What does this document say?" --file "path/to/file.pdf"

# Ingest + Query with custom title
python -m inference.cli infer "What does this document say?" --file "path/to/file.pdf" --title "Doc Title"

# Ingest + Query with cross-doc enabled
python -m inference.cli infer "What does this document say?" --file "path/to/file.pdf" --cross-doc

# Make (from project root)
make infer Q="What does this document say?" FILE="path/to/file.pdf" DOCKER=true
make infer Q="What does this document say?" FILE="path/to/file.pdf" TITLE="Doc Title" CROSS_DOC=true DOCKER=true

# REST API (from any directory)
curl -X POST http://localhost:8000/infer \
  -F "question=What does this document say about RAG systems?" \
  -F "attachment=@path/to/file.pdf" \
  -F "title=Optional Title" \
  -F "cross_doc=true"
```

**What Happens**:
1. Document is ingested → `doc_id` is generated
2. System waits for chunks to be available in the database
3. Query is executed using the direct pipeline with `doc_id` filter
4. If `--cross-doc` is enabled: Two-stage retrieval (see `--cross-doc` flag documentation)
5. Answer is generated with document-specific context

**When to Use**:
- You have a document and want immediate answers
- Simple, straightforward questions
- Fast, deterministic processing

---

### Scenario 5: Ingest + Query (Complex Questions)

**Use Case**: You have a document and want to ingest it and immediately ask complex questions that may require iterative refinement.

**Entry Points**:
- `infer-graph` / `make infer-graph` / `POST /infer-graph`

**Example**:
```bash
# CLI (from deep_rag_backend directory)
cd deep_rag_backend
# Ingest + Query
python -m inference.cli infer-graph "Analyze this document" --file "path/to/file.pdf" --thread-id session-1

# Ingest + Query with custom title
python -m inference.cli infer-graph "Analyze this document" --file "path/to/file.pdf" --title "Doc Title" --thread-id session-1

# Ingest + Query with cross-doc enabled
python -m inference.cli infer-graph "Analyze this document" --file "path/to/file.pdf" --thread-id session-1 --cross-doc

# Make (from project root)
make infer-graph Q="Analyze this document" FILE="path/to/file.pdf" TITLE="Doc Title" DOCKER=true THREAD_ID=session-1

# REST API (from any directory)
curl -X POST http://localhost:8000/infer-graph \
  -F "question=What are the key requirements for this RAG system?" \
  -F "attachment=@path/to/file.pdf" \
  -F "title=Optional Title" \
  -F "thread_id=session-1" \
  -F "cross_doc=true"
```

**What Happens**:
1. Document is ingested → `doc_id` is generated
2. System waits for chunks to be available in the database
3. Query is executed using the LangGraph pipeline with `doc_id` filter
4. Agents can refine queries and retrieve more evidence if confidence is low
5. If `--cross-doc` is enabled: Two-stage retrieval (see `--cross-doc` flag documentation)
6. Answer is generated with document-specific context
7. **Reasoning logs are saved** to `inference/graph/logs/` for SFT training

**When to Use**:
- You have a document and want to perform complex reasoning over it
- Questions that may need iterative refinement
- When you want reasoning logs for future training
- When you need conversation state (`--thread-id`)

---

## `--cross-doc` Flag Scenarios

### Scenario A: Document-Specific with Cross-Document Expansion

**Use Case**: You want to start with a specific document but also find related information across your entire knowledge base.

**Example**:
```bash
# CLI (from deep_rag_backend directory)
cd deep_rag_backend
# Query with doc_id + cross-doc
python -m inference.cli query "What are the requirements?" --doc-id 550e8400-e29b-41d4-a716-446655440000 --cross-doc
```

**What Happens**:
1. **Stage 1**: Retrieves content from the specified `doc_id`
2. **Stage 2**: Uses the original query combined with retrieved content from Stage 1 to search semantically across **all documents**
3. **Merge & Deduplicate**: Results from both stages are combined and deduplicated, with primary chunks prioritized

**When to Use**:
- You have a primary document but want comprehensive answers
- You want to find related information across your knowledge base
- You need both document-specific and cross-document context

---

### Scenario B: General Cross-Document Search

**Use Case**: You want the most comprehensive answer possible from your entire knowledge base.

**Example**:
```bash
# CLI (from deep_rag_backend directory)
cd deep_rag_backend
# Query all documents with cross-doc enabled
python -m inference.cli query "What are the requirements?" --cross-doc
```

**What Happens**:
1. Enhanced cross-document semantic search with better query expansion
2. More comprehensive retrieval across all documents

**When to Use**:
- You're unsure which document contains the answer
- You want the most comprehensive answer possible
- You want better semantic matching across documents

---

## Decision Tree

### Which Entry Point Should I Use?

1. **Do you need to ingest a document?**
   - **Yes** → Use `infer` or `infer-graph`
   - **No** → Use `query` or `query-graph`

2. **Is your question complex and may need iterative refinement?**
   - **Yes** → Use `query-graph` or `infer-graph` (LangGraph pipeline)
   - **No** → Use `query` or `infer` (Direct pipeline)

3. **Do you need conversation state across multiple queries?**
   - **Yes** → Use `query-graph` or `infer-graph` with `--thread-id`
   - **No** → Any entry point works

4. **Do you want reasoning logs for future training?**
   - **Yes** → Use `query-graph` or `infer-graph` (LangGraph pipeline)
   - **No** → Any entry point works

5. **Do you want to search beyond a specific document?**
   - **Yes** → Use `--cross-doc` flag
   - **No** → Don't use `--cross-doc` flag

---

## Best Practices

1. **Use `ingest` for batch workflows**: Pre-populate your knowledge base before querying
2. **Use `query` for simple questions**: Fast, deterministic answers for straightforward questions
3. **Use `query-graph` for complex questions**: When you need iterative refinement and reasoning logs
4. **Use `infer` for immediate answers**: When you have a document and want simple answers quickly
5. **Use `infer-graph` for complex analysis**: When you have a document and need complex reasoning
6. **Store `doc_id` after ingestion**: Save the returned `doc_id` for future queries
7. **Use `--cross-doc` for comprehensive answers**: When you want to search beyond a specific document
8. **Use `--thread-id` for conversations**: When you need to maintain state across multiple queries

---

## Examples by Use Case

### Use Case: Research Assistant

**Scenario**: You're researching a topic across multiple documents.

**Workflow**:
1. Ingest multiple documents using `ingest`
2. Store `doc_id`s for each document
3. Query with `query-graph` and `--cross-doc` to find comprehensive answers
4. Use `--thread-id` to maintain conversation context

**Example**:
```bash
# CLI (from deep_rag_backend directory)
cd deep_rag_backend
# Ingest documents
python -m inference.cli ingest "research_paper_1.pdf"
python -m inference.cli ingest "research_paper_2.pdf"

# Query with cross-doc
python -m inference.cli query-graph "What are the key findings across these papers?" --thread-id research-session --cross-doc
```

---

### Use Case: Document Q&A

**Scenario**: You have a single document and want to ask questions about it.

**Workflow**:
1. Use `infer` or `infer-graph` to ingest and query in one operation
2. Use `--cross-doc` if you want to find related information elsewhere

**Example**:
```bash
# CLI (from deep_rag_backend directory)
cd deep_rag_backend
# Ingest + Query
python -m inference.cli infer "What are the main points?" --file "document.pdf"

# Ingest + Query with cross-doc
python -m inference.cli infer "What are the main points?" --file "document.pdf" --cross-doc
```

---

### Use Case: Knowledge Base Maintenance

**Scenario**: You want to maintain a knowledge base and query it regularly.

**Workflow**:
1. Use `ingest` to pre-populate the knowledge base
2. Store `doc_id`s for future reference
3. Use `query` or `query-graph` to query existing documents
4. Use `inspect` to verify ingestion quality

**Example**:
```bash
# CLI (from deep_rag_backend directory)
cd deep_rag_backend
# Ingest documents
python -m inference.cli ingest "knowledge_base_doc_1.pdf"
python -m inference.cli ingest "knowledge_base_doc_2.pdf"

# Inspect documents
python -m inference.cli inspect --title "knowledge_base_doc_1"

# Query documents
python -m inference.cli query "What is in the knowledge base?" --cross-doc
```

---

For more information on thread tracking and audit logging, see [`THREAD_TRACKING_AND_AUDIT.md`](THREAD_TRACKING_AND_AUDIT.md).

