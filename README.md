# 🧠 Deep RAG — Agentic Retrieval-Augmented Generation Pipeline

Deep RAG is a modular, production-ready Retrieval-Augmented Generation (RAG) system for querying and reasoning over PDFs (text + images).  
It combines deterministic PDF parsing with hybrid (lexical + vector) retrieval, cross-encoder reranking, and an agentic multi-stage reasoning loop inspired by "deep-thinking RAG" architectures.

---

# 📂 Directory Structure

```bash
deep_rag/
├── inference/                    # Inference pipeline and API
│   ├── cli.py                    # Typer CLI interface matching FastAPI service routes
│   ├── service.py                # FastAPI REST API entrypoint with all endpoints
│   ├── agents/                   # Direct pipeline: modularized agent components
│   │   ├── __init__.py
│   │   ├── pipeline.py           # Main pipeline orchestrator
│   │   ├── state.py              # State TypedDict definition
│   │   ├── constants.py          # Pipeline constants (MAX_ITERS, THRESH)
│   │   ├── planner.py            # Planning agent
│   │   ├── retriever.py          # Retrieval agent
│   │   ├── compressor.py         # Compression agent
│   │   ├── critic.py             # Critic agent
│   │   └── synthesizer.py        # Synthesis agent
│   ├── llm/                      # LLM provider interface (modularized)
│   │   ├── __init__.py
│   │   ├── wrapper.py            # Unified LLM interface
│   │   ├── config.py             # LLM configuration and env vars
│   │   └── providers/            # LLM provider implementations
│   │       ├── __init__.py
│   │       └── gemini.py         # Google Gemini provider
│   ├── graph/                    # LangGraph pipeline
│   │   ├── graph.py              # Legacy wrapper (backward compatibility)
│   │   ├── graph_wrapper.py      # Wrapper for LangGraph pipeline with logging
│   │   ├── graph_viz.py          # Graph visualization export (PNG/Mermaid)
│   │   ├── state.py              # GraphState TypedDict definition
│   │   ├── constants.py          # Graph constants (MAX_ITERS, THRESH)
│   │   ├── builder.py            # Graph builder and compiler
│   │   ├── routing.py            # Conditional routing logic
│   │   ├── agent_logger.py       # Agent logging for SFT training
│   │   └── nodes/                # Individual graph nodes
│   │       ├── __init__.py
│   │       ├── planner.py
│   │       ├── retriever.py
│   │       ├── compressor.py
│   │       ├── critic.py
│   │       ├── refine_retrieve.py
│   │       └── synthesizer.py
│   ├── commands/                 # CLI command modules
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── query.py
│   │   ├── query_graph.py
│   │   ├── infer.py
│   │   ├── infer_graph.py
│   │   ├── inspect.py
│   │   ├── graph.py
│   │   ├── health.py
│   │   └── test.py
│   ├── routes/                   # FastAPI route modules
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── ask.py
│   │   ├── ask_graph.py
│   │   ├── infer.py
│   │   ├── infer_graph.py
│   │   ├── diagnostics.py
│   │   ├── graph_export.py
│   │   ├── health.py
│   │   └── models.py
│   └── samples/                  # Sample PDFs + Images for testing
│
├── ingestion/                    # Document ingestion pipeline
│   ├── ingest.py                 # Main PDF ingestion orchestrator
│   ├── ingest_text.py            # Plain text file ingestion
│   ├── ingest_image.py           # Image file ingestion (PNG, JPEG) with OCR
│   ├── ingest_unified.py         # Unified ingestion interface
│   ├── pdf_extract.py            # PDF text extraction with OCR fallback
│   ├── chunking.py               # Semantic chunking logic
│   ├── title_extract.py          # Document title extraction
│   ├── db_ops/                   # Database operations (modularized)
│   │   ├── __init__.py
│   │   ├── document.py           # Document upsert operations
│   │   └── chunks.py             # Chunk upsert operations
│   └── embeddings/               # Multi-modal embedding system (modularized)
│       ├── __init__.py
│       ├── model.py              # CLIP model loading and configuration
│       ├── text.py               # Text embedding generation
│       ├── image.py              # Image embedding generation
│       ├── multimodal.py         # Multi-modal embedding utilities
│       ├── batch.py              # Batch embedding operations
│       └── utils.py              # Embedding utilities
│
├── retrieval/                    # Hybrid retrieval system
│   ├── retrieval.py              # Main hybrid retrieval orchestrator
│   ├── sanitize.py               # Query sanitization for SQL
│   ├── mmr.py                    # Maximal Marginal Relevance (MMR) diversity
│   ├── vector_utils.py           # Vector utility functions
│   ├── wait.py                   # Wait for chunks to be available
│   ├── db_utils.py               # Centralized database connection utilities
│   ├── sql/                      # SQL query generation (modularized)
│   │   ├── __init__.py
│   │   ├── hybrid.py             # Hybrid SQL query with doc_id filter
│   │   └── exclusion.py          # Hybrid SQL query with doc_id exclusion
│   ├── reranker/                 # Cross-encoder reranking (modularized)
│   │   ├── __init__.py
│   │   ├── model.py              # Reranker model loading
│   │   └── rerank.py             # Reranking logic
│   ├── stages/                   # Two-stage retrieval (modularized)
│   │   ├── __init__.py
│   │   ├── stage_one.py          # Stage 1: Primary retrieval
│   │   ├── stage_two.py          # Stage 2: Cross-document retrieval
│   │   └── merge.py              # Merge and deduplicate results
│   ├── diagnostics/              # Diagnostic tools (modularized)
│   │   ├── __init__.py
│   │   ├── inspect.py            # Document inspection
│   │   └── report.py             # Inspection report generation
│   ├── thread_tracking/          # Thread tracking and audit logging (modularized)
│   │   ├── __init__.py
│   │   ├── log.py                # Log thread interactions
│   │   ├── get.py                # Retrieve thread interactions
│   │   └── update.py             # Update thread interactions
│   └── diagnostics.py            # Legacy wrapper (backward compatibility)
│
├── tests/                        # Comprehensive test suite
│   ├── __init__.py
│   ├── conftest.py               # Pytest configuration and fixtures
│   ├── unit/                     # Unit tests
│   │   ├── __init__.py
│   │   ├── test_retrieval_sql.py
│   │   ├── test_retrieval_sanitize.py
│   │   ├── test_retrieval_mmr_basic.py
│   │   ├── test_retrieval_mmr_diversity.py
│   │   ├── test_retrieval_vector_utils.py
│   │   ├── test_retrieval_wait.py
│   │   ├── test_retrieval_merge.py
│   │   ├── test_llm_wrapper.py
│   │   ├── test_llm_providers_gemini.py
│   │   └── test_embeddings_text.py  # Embedding model and text embedding tests
│   └── integration/              # Integration tests
│       ├── __init__.py
│       ├── test_llm_providers.py
│       ├── test_llm_providers_gemini.py
│       ├── test_llm_providers_openai.py
│       ├── test_llm_providers_ollama.py
│       └── test_database_schema.py  # Database schema verification tests
│
├── vector_db/                    # Database schema and migrations
│   ├── schema_multimodal.sql     # Multi-modal schema for CLIP embeddings (768 dims)
│   ├── migration_add_multimodal.sql  # Migration for multi-modal support
│   ├── migration_add_thread_tracking.sql  # Migration for thread tracking table
│   ├── ingestion_schema.sql      # Legacy schema (for reference)
│   └── docker-compose.yml        # Stand-alone DB service (pgvector)
│
├── scripts/                      # Docker and deployment scripts
│   └── entrypoint.sh             # Docker container entrypoint (runs tests on startup if enabled)
│
├── md_guides/                    # Documentation guides
│   ├── EMBEDDING_OPTIONS.md      # Embedding model options and recommendations
│   ├── LLM_SETUP.md              # LLM provider setup guide
│   ├── PROJECT_ASSESSMENT.md     # Project assessment requirements
│   ├── RESET_DB.md               # Database reset instructions
│   ├── SETUP_GUIDE.md            # Detailed setup guide
│   ├── ENTRY_POINTS_AND_SCENARIOS.md  # Entry point scenarios and use cases
│   └── THREAD_TRACKING_AND_AUDIT.md   # Thread tracking and audit logging
│
├── .env.example                  # Example environment variables (copy to .env and fill in your values)
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project metadata, dependencies, and pytest config
├── makefile                      # Convenience make commands for common tasks
├── Dockerfile                    # API container definition
└── docker-compose.yml            # Root-level orchestration (API + DB)
```

---

# 🎯 Entry Points

The Deep RAG system provides multiple entry points (CLI, Make, TOML, REST API) for different use cases:

## Entry Point Mapping Table

| CLI Command | Make Script | REST Endpoint | Pipeline | Purpose |
|------------|-------------|---------------|----------|---------|
| `ingest` | `make cli-ingest` | `POST /ingest` | Direct | **Ingestion only**: Embeds documents into vector DB without querying. Use when you want to pre-populate your knowledge base. |
| `query` | `make query` | `POST /ask` | Direct (`inference/agents/pipeline.py`) | **Query only (direct pipeline)**: Fast, deterministic pipeline for simple queries. No conditional routing. Best for straightforward questions. Supports `--doc-id` and `--cross-doc` flags. |
| `query-graph` | `make query-graph` | `POST /ask-graph` | LangGraph | **Query only (LangGraph)**: Agentic pipeline with conditional routing. Agents can refine queries and retrieve more evidence if confidence is low. Best for complex questions requiring iterative reasoning. Supports `--doc-id`, `--thread-id`, and `--cross-doc` flags. |
| `infer` | `make infer` | `POST /infer` | Direct (`inference/agents/pipeline.py`) | **Ingest + Query (direct pipeline)**: Combined ingestion and querying in one operation. Use when you have a document and want immediate answers with fast, deterministic processing. Supports `--file`, `--title`, and `--cross-doc` flags. |
| `infer-graph` | `make infer-graph` | `POST /infer-graph` | LangGraph | **Ingest + Query (LangGraph)**: Combined ingestion with agentic reasoning. Best when you need to ingest and then perform complex reasoning over the new content. Supports `--file`, `--title`, `--thread-id`, and `--cross-doc` flags. |
| `health` | N/A | `GET /health` | N/A | **Health check**: Verifies database connectivity and service availability. |
| `graph` | `make graph` | `GET /graph` | N/A | **Graph export**: Exports LangGraph pipeline visualization as PNG or Mermaid diagram. Useful for understanding the agentic flow. |
| `inspect` | `make inspect` | `GET /diagnostics/document` | N/A | **Document diagnostics**: Inspects what chunks and pages are stored for a document. Shows page distribution, chunk counts, and sample text. Essential for debugging ingestion and retrieval issues. |
| `test` | `make test` | N/A | N/A | **Testing**: Run all tests (unit + integration). Supports `--docker` flag. |
| `test unit` | `make unit-tests` | N/A | N/A | **Unit tests only**: Run unit tests for individual modules. Supports `--docker` flag. |
| `test integration` | `make integration-tests` | N/A | N/A | **Integration tests only**: Run integration tests for end-to-end workflows. Supports `--docker` flag. |

### Pipeline Comparison

- **Direct Pipeline** (`inference/agents/pipeline.py`): Linear execution, faster, deterministic. Best for simple queries.
- **LangGraph Pipeline** (`inference/graph/builder.py`): Conditional routing, agents can refine queries, iterative retrieval. Best for complex questions requiring multi-step reasoning.

For detailed scenarios and use cases for each entry point, see [`md_guides/ENTRY_POINTS_AND_SCENARIOS.md`](md_guides/ENTRY_POINTS_AND_SCENARIOS.md).

---

# 🚀 Features

| Layer | Description |
|-------|--------------|
| **Ingestion** | Multi-modal ingestion: PDFs (text + OCR), plain text files, and images (PNG/JPEG). Extracts text using PyMuPDF (`fitz`) with OCR fallback (`pytesseract`). Chunks and embeds with **CLIP-ViT-L/14** (`sentence-transformers/clip-ViT-L-14`) into unified **768-dimensional vectors** in Postgres + pgvector. **Upgraded from ViT-B/32 (512 dims)** for better semantic representation. |
| **Retrieval** | Hybrid search combining pg_trgm (BM25-style) lexical scores + vector similarity (cosine distance). Reranked by a cross-encoder (`bge-reranker-base`). Supports multi-modal queries (text + images). Dynamically supports 512 or 768 dimensional embeddings. **Supports two-stage retrieval with `--cross-doc` flag** for cross-document semantic search. |
| **Agentic Loop** | Two pipeline options: (1) **Direct pipeline** (`inference/agents/pipeline.py`): Linear execution (plan → retrieve → compress → reflect → synthesize). (2) **LangGraph pipeline** (`inference/graph/builder.py`): Conditional routing with iterative refinement - agents can refine queries and retrieve more evidence when confidence < threshold. **Includes comprehensive logging** to CSV/TXT for future SFT training. |
| **LLM Integration** | Currently using **Google Gemini** for agentic reasoning. **Recommended models**: `gemini-1.5-flash` (1M token context, best balance) or `gemini-2.0-flash` (latest, improved reasoning). Alternative: `gemini-2.5-flash-lite` for faster, lightweight processing. Code structure supports future integration with OpenAI, Ollama, and LLaVA. |
| **Multi-modal Support** | Unified embedding space for text and images using CLIP-ViT-L/14 (768 dims), enabling semantic search across different content types with better representation than ViT-B/32. |
| **Cross-Document Retrieval** | **NEW**: `--cross-doc` flag enables two-stage retrieval. When `doc_id` is provided: Stage 1 retrieves from the specified document, Stage 2 uses combined query (original + retrieved content) for semantic search across all documents. When no `doc_id`: Enables general cross-document semantic search. Results are merged and deduplicated. |
| **Document Context** | Document ID (`doc_id`) tracking throughout the pipeline. Enables document-specific filtering, context-aware planning and synthesis, and automatic document identification from retrieved chunks. |
| **Thread Tracking** | **NEW**: Comprehensive audit logging via `thread_tracking` table. Tracks user interactions, thread sessions, document retrievals, pipeline states, and entry points for SFT/RLHF training and analysis. See [`md_guides/THREAD_TRACKING_AND_AUDIT.md`](md_guides/THREAD_TRACKING_AND_AUDIT.md). |
| **Reasoning Logs** | All agentic reasoning steps are logged to `inference/graph/logs/` in both CSV (for training) and TXT (for presentations). Captures queries, plans, retrievals, confidence scores, and refinements for future SFT model training. |
| **Modular Architecture** | Fully modularized codebase with focused modules for agents, LLM providers, retrieval stages, embeddings, database operations, and diagnostics. Improves legibility, testing, and context switching. |
| **Comprehensive Testing** | Unit tests for all modules and integration tests for LLM providers and end-to-end workflows. Tests can be run via CLI, Make, TOML, or direct Pytest. |
| **Microservice Ready** | FastAPI REST interface with comprehensive endpoints for ingestion, querying (direct and LangGraph), and health checks. |
| **CLI Ready** | Typer CLI matching all REST endpoints for easy local development and testing. |
| **Containerized DB** | pgvector/pg16 Docker image with automatic schema init via mounted SQL file. Support for fresh database starts and migrations. |

---

# 🚩 Flags and Options

## `--cross-doc` Flag

The `--cross-doc` flag enables **cross-document retrieval**, allowing the system to search beyond a single specified document for more comprehensive answers.

### Behavior

#### When `doc_id` is Provided

**Without `--cross-doc`**: Retrieval is strictly limited to the specified `doc_id` only.

**With `--cross-doc`**: Performs **two-stage retrieval**:
1. **Stage 1 (Primary)**: Retrieves content from the specified `doc_id`
2. **Stage 2 (Cross-Document)**: Uses the original query combined with retrieved content from Stage 1 to formulate a semantic search query across **all documents** (including the primary `doc_id` for semantic search, but prioritizing primary results)
3. **Merge & Deduplicate**: Results from both stages are combined and deduplicated, with primary chunks prioritized

**Use Case**: When you want to start with a specific document but also find related information across your entire knowledge base.

#### When No `doc_id` is Provided

**Without `--cross-doc`**: Standard retrieval searches across all documents.

**With `--cross-doc`**: Enables enhanced cross-document semantic search with better query expansion and semantic matching.

**Use Case**: When you want the most comprehensive answer possible from your entire knowledge base.

### Examples

#### CLI
```bash
# Query with doc_id only (strict filtering)
python inference/cli.py query "What are the requirements?" --doc-id 550e8400-e29b-41d4-a716-446655440000

# Query with doc_id + cross-doc (two-stage retrieval)
python inference/cli.py query "What are the requirements?" --doc-id 550e8400-e29b-41d4-a716-446655440000 --cross-doc

# Query all documents with cross-doc enabled
python inference/cli.py query "What are the requirements?" --cross-doc

# Ingest + Query with cross-doc
python inference/cli.py infer "What does this document say?" --file "path/to/file.pdf" --cross-doc
```

#### Make
```bash
# Query with doc_id + cross-doc
make query Q="What are the requirements?" DOC_ID=550e8400-e29b-41d4-a716-446655440000 CROSS_DOC=true DOCKER=true

# Query all documents with cross-doc
make query Q="What are the requirements?" CROSS_DOC=true DOCKER=true

# Ingest + Query with cross-doc
make infer Q="What does this document say?" FILE="path/to/file.pdf" CROSS_DOC=true DOCKER=true
```

#### REST API
```bash
# Query with doc_id + cross-doc
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the requirements?", "doc_id": "550e8400-e29b-41d4-a716-446655440000", "cross_doc": true}'

# Query all documents with cross-doc
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the requirements?", "cross_doc": true}'
```

### When to Use `--cross-doc`

- ✅ **Use when**: You want comprehensive answers that may span multiple documents
- ✅ **Use when**: You have a primary document but want to find related information elsewhere
- ✅ **Use when**: You're unsure which document contains the answer
- ❌ **Don't use when**: You need strict document-specific answers
- ❌ **Don't use when**: You want faster, more focused retrieval from a single document

## Other Flags

### `--doc-id` (Document ID)
Filters retrieval to a specific document. See [Document ID (`doc_id`) and Context Reasoning](#-document-id-doc_id-and-context-reasoning) section.

### `--thread-id` (Thread ID)
Used with LangGraph pipeline to maintain conversation state across multiple queries. Default: `"default"`.

### `--title` (Document Title)
Custom title for documents during ingestion. If not provided, extracted from document content.

### `--file` (File Path)
File path for ingestion (PDF, TXT, PNG, JPEG). Used with `infer` and `infer-graph` commands.

### `--docker` (Docker Flag)
Run commands inside Docker container. Used with Make scripts and CLI test commands.

### `--verbose` / `--quiet` (Verbosity)
Control output verbosity. Used with CLI test commands.

---

# 📄 Document ID (`doc_id`) and Context Reasoning

## Overview

Deep RAG uses **Document IDs (`doc_id`)** to enable document-specific retrieval and context-aware reasoning. Every chunk in the vector database is linked to its source document via a `doc_id` UUID, allowing the system to:

- **Filter retrieval** to specific documents when needed
- **Track provenance** of retrieved information
- **Provide document context** to the LLM for better reasoning
- **Identify document sources** from retrieved chunks when querying without ingestion

## Database Schema

The `doc_id` is stored in the PostgreSQL `chunks` table:

```sql
CREATE TABLE chunks (
  chunk_id    UUID PRIMARY KEY,
  doc_id      UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
  page_start  INT,
  page_end    INT,
  text        TEXT NOT NULL,
  emb         vector(768),  -- pgvector embedding
  ...
);
```

**Key Points:**
- `doc_id` is a UUID that references the `documents` table
- Every chunk is linked to its source document via `doc_id`
- `ON DELETE CASCADE` ensures chunks are deleted when a document is removed
- `doc_id` is included in all retrieval queries and results

## How `doc_id` Flows Through the System

### During Ingestion

When a document is ingested (PDF, TXT, PNG, JPEG), the system:
1. Creates a new document record in the `documents` table → generates `doc_id`
2. Chunks the document content
3. Embeds each chunk
4. Inserts chunks into the `chunks` table with the `doc_id` reference
5. **Returns the `doc_id`** to the caller

### During Retrieval

When querying, the system:
1. **If `doc_id` is provided**: Filters retrieval to chunks from that specific document
2. **If `doc_id` is not provided**: Searches across all documents in the knowledge base
3. **If `doc_id` is not provided but chunks are retrieved**: The synthesizer can identify `doc_id` from retrieved chunks (if all chunks come from one document)

### During Synthesis

The synthesizer uses `doc_id` context to:
- Include document-specific context in the LLM prompt
- Provide better reasoning about which document the answer is based on
- Log which document(s) were used for the answer

## Usage Scenarios

For detailed scenarios and examples, see [`md_guides/ENTRY_POINTS_AND_SCENARIOS.md`](md_guides/ENTRY_POINTS_AND_SCENARIOS.md).

---

# 🧩 Prerequisites

- **Python ≥ 3.11** (required for Google Gemini support due to 3.10 support deprecation in 2026)  
- **Docker & Docker Compose**  
- **(Optional) Tesseract OCR + Poppler** (for scanned PDFs)

## Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr poppler-utils -y

# macOS
brew install tesseract poppler
```

---

# 🚀 Quick Start

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/omoral02/deep_rag.git
cd deep_rag
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the project root by copying the example file:
```bash
# Copy the example file
cp .env.example .env

# Then edit .env with your actual credentials and API keys
# Never commit .env to git - it contains sensitive information
```

The `.env.example` file contains all required environment variables with sample values. See [`md_guides/ENVIRONMENT_SETUP.md`](md_guides/ENVIRONMENT_SETUP.md) for detailed configuration options.

**Required Variables:**
- **Database**: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASS`, `DB_NAME`
- **LLM**: `LLM_PROVIDER`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `LLM_TEMPERATURE`
- **Embeddings**: `CLIP_MODEL`, `EMBEDDING_DIM`

**Optional Variables:**
- **Startup Tests**: `RUN_TESTS_ON_STARTUP` (set to `true` to run database schema tests on container startup)

### 5. Start Services
```bash
# Option 1: Start full stack (API + DB)
make up              # Or: docker compose up -d --build

# Option 2: Start and run tests automatically
make up-and-test     # Starts services, then runs all tests to verify setup

# Option 3: Start DB only (for local development)
make db-up           # Or: cd vector_db && docker-compose up -d
```

**Note:** The Docker container uses an entrypoint script (`scripts/entrypoint.sh`) that:
- Optionally runs database schema tests on startup if `RUN_TESTS_ON_STARTUP=true` is set in your `.env`
- Starts the FastAPI server on port 8000
- Provides health check endpoint at `/health` that verifies database connection and required tables

### 6. Verify Services
```bash
# Check running containers
docker ps
# Should show: deep_rag_pgvector and deep_rag_api running

# Check API health (verifies database connection and schema)
curl http://localhost:8000/health
# Should return: {"ok": true, "status": "healthy", "database": "connected", "tables": ["chunks", "documents", "thread_tracking"]}
```

---

# 📖 Usage Examples

All entry points are organized below. Choose the method that best fits your workflow.

## Via CLI (Command Line Interface)

### Run Locally
```bash
# Ingest a document
python inference/cli.py ingest "path/to/file.pdf"
python inference/cli.py ingest "path/to/file.pdf" --title "Custom Title"

# Query existing documents
python inference/cli.py query "What are the main sections?"
python inference/cli.py query "What are the requirements?" --doc-id 550e8400-e29b-41d4-a716-446655440000
python inference/cli.py query "What are the requirements?" --doc-id 550e8400-e29b-41d4-a716-446655440000 --cross-doc
python inference/cli.py query-graph "What are the requirements?" --thread-id session-1
python inference/cli.py query-graph "What are the requirements?" --doc-id 550e8400-e29b-41d4-a716-446655440000 --thread-id session-1 --cross-doc

# Ingest + Query in one command
python inference/cli.py infer "What does this document say?" --file "path/to/file.pdf"
python inference/cli.py infer "What does this document say?" --file "path/to/file.pdf" --cross-doc
python inference/cli.py infer-graph "Analyze this document" --file "path/to/file.pdf" --title "Doc Title" --thread-id session-1

# Inspect stored chunks and pages (debugging)
python inference/cli.py inspect --title "Document Title"
python inference/cli.py inspect --doc-id your-doc-id-here
python inference/cli.py inspect  # List all documents

# Export graph visualization
python inference/cli.py graph --out deep_rag_graph.png

# Health check
python inference/cli.py health

# Testing
python inference/cli.py test all          # Run all tests (unit + integration)
python inference/cli.py test unit         # Run unit tests only
python inference/cli.py test integration   # Run integration tests only
python inference/cli.py test all --docker  # Run tests inside Docker container
```

### Run Inside Docker
```bash
# Start the stack
docker compose up -d --build

# Ingest a document (inside Docker)
docker compose exec api python -m inference.cli ingest "inference/samples/file.pdf"
docker compose exec api python -m inference.cli ingest "inference/samples/file.pdf" --title "Custom Title"

# Query documents (inside Docker)
docker compose exec api python -m inference.cli query "What are the requirements?"
docker compose exec api python -m inference.cli query "What are the requirements?" --doc-id 550e8400-e29b-41d4-a716-446655440000 --cross-doc
docker compose exec api python -m inference.cli query-graph "Complex question" --thread-id session-1

# Inspect stored chunks (inside Docker)
docker compose exec api python -m inference.cli inspect --title "Document Title"
docker compose exec api python -m inference.cli inspect --doc-id your-doc-id-here

# Run tests (inside Docker)
docker compose exec api python -m inference.cli test all
docker compose exec api python -m inference.cli test unit
docker compose exec api python -m inference.cli test integration
```

---

## Via Make Scripts

**Install Make:**
```bash
# Check if installed
make --version

# Install if needed:
# Ubuntu/Debian/WSL: sudo apt update && sudo apt install -y make build-essential
# macOS: xcode-select --install
# Windows: choco install make
```

### Common Commands
```bash
# Start/stop services
make up              # Start full stack (API + DB)
make down            # Stop and remove containers/volumes
make logs            # Tail API + DB logs
make rebuild         # Rebuild API image and restart

# DB-only operations
make db-up           # Start DB only
make db-down         # Stop DB-only stack

# Ingest documents
make cli-ingest FILE="path/to/file.pdf" DOCKER=true
make cli-ingest FILE="path/to/file.pdf" DOCKER=true TITLE="Custom Title"

# Query documents
# Query all documents
make query Q="Your question here" DOCKER=true
make query Q="Your question here" CROSS_DOC=true DOCKER=true
make query-graph Q="Complex question" DOCKER=true THREAD_ID=session-1

# Query specific document by doc_id (Scenario 3)
make query Q="What are the requirements?" DOCKER=true DOC_ID=550e8400-e29b-41d4-a716-446655440000
make query Q="What are the requirements?" DOCKER=true DOC_ID=550e8400-e29b-41d4-a716-446655440000 CROSS_DOC=true
make query-graph Q="What are the requirements?" DOCKER=true THREAD_ID=session-1 DOC_ID=550e8400-e29b-41d4-a716-446655440000 CROSS_DOC=true

# Ingest + Query
make infer Q="Your question" FILE="path/to/file.pdf" TITLE="Title" DOCKER=true
make infer Q="Your question" FILE="path/to/file.pdf" TITLE="Title" CROSS_DOC=true DOCKER=true
make infer-graph Q="Your question" FILE="path/to/file.pdf" TITLE="Title" DOCKER=true THREAD_ID=session-1

# Diagnostics
make inspect TITLE="Document Title" DOCKER=true
make inspect DOC_ID=your-doc-id-here DOCKER=true
make inspect DOCKER=true  # List all documents

# Graph visualization
make graph OUT=deep_rag_graph.png DOCKER=true

# Testing
make test                  # Run all tests (unit + integration)
make unit-tests           # Run unit tests only
make integration-tests    # Run integration tests only
make test DOCKER=true     # Run tests inside Docker container
make unit-tests DOCKER=true
make integration-tests DOCKER=true
```

---

## Via TOML Script (pyproject.toml)

**Install project as package:**
```bash
pip install -e .
```

**Then use:**
```bash
# Ingest
deep-rag ingest path/to/file.pdf
deep-rag ingest path/to/file.pdf --title "Custom Title"

# Query
deep-rag query "What are the main sections?"
deep-rag query "What are the requirements?" --doc-id 550e8400-e29b-41d4-a716-446655440000
deep-rag query "What are the requirements?" --doc-id 550e8400-e29b-41d4-a716-446655440000 --cross-doc
deep-rag query-graph "What are the requirements?" --thread-id session-1

# Ingest + Query
deep-rag infer "Question" --file path/to/file.pdf
deep-rag infer "Question" --file path/to/file.pdf --cross-doc
deep-rag infer-graph "Question" --file path/to/file.pdf --title "Title" --thread-id session-1

# Diagnostics
deep-rag inspect --title "Document Title"
deep-rag inspect --doc-id your-doc-id-here
deep-rag inspect  # List all documents

# Graph visualization
deep-rag graph --out deep_rag_graph.png

# Testing
deep-rag test all          # Run all tests (unit + integration)
deep-rag test unit         # Run unit tests only
deep-rag test integration  # Run integration tests only
deep-rag test all --docker # Run tests inside Docker container
```

---

## Via REST API

### Start the API Server
```bash
# Start full stack
docker compose up -d --build

# Or run locally (requires dependencies installed)
uvicorn inference.service:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### POST /ingest (File Upload)
```bash
curl -X POST http://localhost:8000/ingest \
  -F "attachment=@path/to/file.pdf" \
  -F "title=Optional Document Title"
```

#### POST /ask (Query - Direct Pipeline)
```bash
# Query all documents
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the document about?"}'

# Query specific document by doc_id
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the document about?", "doc_id": "550e8400-e29b-41d4-a716-446655440000"}'

# Query with cross-doc enabled
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the document about?", "doc_id": "550e8400-e29b-41d4-a716-446655440000", "cross_doc": true}'
```

#### POST /ask-graph (Query - LangGraph Pipeline)
```bash
# Query all documents
curl -X POST http://localhost:8000/ask-graph \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the specific requirements?", "thread_id": "session-1"}'

# Query specific document by doc_id
curl -X POST http://localhost:8000/ask-graph \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the specific requirements?", "thread_id": "session-1", "doc_id": "550e8400-e29b-41d4-a716-446655440000"}'

# Query with cross-doc enabled
curl -X POST http://localhost:8000/ask-graph \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the specific requirements?", "thread_id": "session-1", "doc_id": "550e8400-e29b-41d4-a716-446655440000", "cross_doc": true}'
```

#### POST /infer (Ingest + Query - Direct)
```bash
curl -X POST http://localhost:8000/infer \
  -F "question=What does this document say about RAG systems?" \
  -F "attachment=@path/to/file.pdf" \
  -F "title=Optional Title"

# With cross-doc enabled
curl -X POST http://localhost:8000/infer \
  -F "question=What does this document say about RAG systems?" \
  -F "attachment=@path/to/file.pdf" \
  -F "title=Optional Title" \
  -F "cross_doc=true"
```

#### POST /infer-graph (Ingest + Query - LangGraph)
```bash
curl -X POST http://localhost:8000/infer-graph \
  -F "question=What are the key requirements for this RAG system?" \
  -F "attachment=@path/to/file.pdf" \
  -F "title=Optional Title" \
  -F "thread_id=session-1"
```

#### GET /health
```bash
curl http://localhost:8000/health
```

#### GET /graph
```bash
curl "http://localhost:8000/graph?out=deep_rag_graph.png" -o deep_rag_graph.png
```

#### GET /diagnostics/document
```bash
# Inspect by document title (partial match)
curl "http://localhost:8000/diagnostics/document?doc_title=NYMBL"

# Inspect by document ID (UUID)
curl "http://localhost:8000/diagnostics/document?doc_id=your-doc-id-here"

# List all documents (if no params provided)
curl "http://localhost:8000/diagnostics/document"
```

---

# 🧠 Multi-Modal Embedding Model

## Current Model: CLIP-ViT-L/14

Deep RAG uses **CLIP-ViT-L/14** (`sentence-transformers/clip-ViT-L-14`) for multi-modal embeddings, providing a unified vector space for both text and images.

### Model Specifications

| Property | CLIP-ViT-L/14 (Current) | CLIP-ViT-B/32 (Legacy) |
|----------|-------------------------|------------------------|
| **Embedding Dimensions** | **768** | 512 |
| **Max Token Length** | 77 tokens | 77 tokens |
| **Performance** | **Better semantic representation** | Faster, lower memory |
| **Model Size** | ~400MB | ~150MB |
| **Use Case** | Production, high-quality retrieval | Development, resource-constrained |

### Why CLIP-ViT-L/14?

1. **Higher Dimensions (768 vs 512)**: More dimensional space = better semantic representation and retrieval accuracy
2. **Multi-Modal**: Embeds text and images in the same vector space, enabling true multi-modal search
3. **Open-Source & Local**: Runs entirely locally without API dependencies
4. **pgvector Compatible**: 768 dimensions well within pgvector's 2,000 dimension limit
5. **Production Ready**: Better performance for real-world RAG applications

### Model Configuration

Set via environment variables in `.env`:

```bash
# Use CLIP-ViT-L/14 (768 dims, recommended)
CLIP_MODEL=sentence-transformers/clip-ViT-L-14
EMBEDDING_DIM=768

# Or use CLIP-ViT-B/32 (512 dims, faster)
# CLIP_MODEL=sentence-transformers/clip-ViT-B-32
# EMBEDDING_DIM=512
```

For detailed embedding options and selection rationale, see [`md_guides/EMBEDDING_OPTIONS.md`](md_guides/EMBEDDING_OPTIONS.md).

---

# 🤖 LLM Model Recommendations

## Gemini Model Selection

Deep RAG uses Google Gemini for agentic reasoning. **Recommended models** (in order of preference):

### 1. **gemini-1.5-flash** (Recommended)
- **Context Window**: 1 million tokens
- **Speed**: Fast (optimized for throughput)
- **Quality**: Excellent reasoning and instruction following
- **Use Case**: Production RAG applications with large documents
- **Cost**: Cost-effective for high-volume usage

### 2. **gemini-2.0-flash** (Latest)
- **Context Window**: 1 million tokens
- **Speed**: Fast with improved performance
- **Quality**: Latest improvements in reasoning and multi-turn conversations
- **Use Case**: Production with latest capabilities
- **Note**: Newer model, may have different behavior

### 3. **gemini-2.5-flash-lite** (Lightweight)
- **Context Window**: Limited (check current specs)
- **Speed**: Very fast
- **Quality**: Good for simple queries
- **Use Case**: Development, cost-constrained environments, simple Q&A
- **Limitation**: May struggle with complex multi-step reasoning

### Configuration

Set in `.env`:
```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-1.5-flash  # Recommended
LLM_TEMPERATURE=0.2
```

For detailed LLM setup and provider selection rationale, see [`md_guides/LLM_SETUP.md`](md_guides/LLM_SETUP.md).

---

# 🧪 Testing

Deep RAG includes comprehensive unit and integration tests for all modules, agentic pipelines, ingestion pipeline, wrapper functions, logger functions, and graph functions.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual modules
│   ├── test_retrieval_*.py  # Retrieval module tests
│   ├── test_llm_*.py        # LLM provider tests
│   ├── test_embeddings_text.py  # Embedding model and text embedding tests
│   └── ...
├── integration/             # Integration tests
│   ├── test_llm_providers_*.py  # LLM provider integration tests
│   ├── test_database_schema.py  # Database schema verification tests
│   └── ...
└── conftest.py             # Pytest configuration
```

## Test Commands

### Via Make Scripts

**Run all tests (unit + integration):**
```bash
make test                  # Run all tests locally
make test DOCKER=true      # Run all tests inside Docker container
```

**Run unit tests only:**
```bash
make unit-tests           # Run unit tests locally
make unit-tests DOCKER=true  # Run unit tests inside Docker container
```

**Run integration tests only:**
```bash
make integration-tests    # Run integration tests locally
make integration-tests DOCKER=true  # Run integration tests inside Docker container
```

### Via CLI (Command Line Interface)

**Run all tests:**
```bash
# Run locally
python inference/cli.py test all
python inference/cli.py test all --verbose  # Verbose output
python inference/cli.py test all --quiet    # Quiet output

# Run inside Docker
python inference/cli.py test all --docker
```

**Run unit tests only:**
```bash
# Run locally
python inference/cli.py test unit
python inference/cli.py test unit --verbose

# Run inside Docker
python inference/cli.py test unit --docker
```

**Run integration tests only:**
```bash
# Run locally
python inference/cli.py test integration
python inference/cli.py test integration --verbose

# Run inside Docker
python inference/cli.py test integration --docker
```

### Via TOML Script (pyproject.toml)

**Install project as package:**
```bash
pip install -e .
```

**Then use:**
```bash
# Run all tests
deep-rag test all
deep-rag test all --docker
deep-rag test all --verbose

# Run unit tests only
deep-rag test unit
deep-rag test unit --docker
deep-rag test unit --verbose

# Run integration tests only
deep-rag test integration
deep-rag test integration --docker
deep-rag test integration --verbose
```

### Via Direct Pytest

**Run all tests:**
```bash
# Run locally
pytest tests/ -v

# Run inside Docker
docker compose exec api python -m pytest tests/ -v
```

**Run unit tests only:**
```bash
# Run locally
pytest tests/unit/ -v

# Run inside Docker
docker compose exec api python -m pytest tests/unit/ -v
```

**Run integration tests only:**
```bash
# Run locally
pytest tests/integration/ -v

# Run inside Docker
docker compose exec api python -m pytest tests/integration/ -v
```

## Test Coverage

### Unit Tests

Unit tests cover:
- **Retrieval modules**: `retrieval/sql/`, `retrieval/reranker/`, `retrieval/stages/`, `retrieval/sanitize.py`, `retrieval/mmr.py`, `retrieval/vector_utils.py`, `retrieval/wait.py`
  - SQL query generation and sanitization
  - Maximal Marginal Relevance (MMR) diversity
  - Vector similarity calculations
  - Two-stage retrieval merge and deduplication
  - Chunk availability waiting logic
- **Embedding modules**: `ingestion/embeddings/model.py`, `ingestion/embeddings/text.py`
  - CLIP model initialization and validation
  - Text embedding generation with tokenization
  - Model lazy loading and caching
  - Error handling for invalid models and missing environment variables
- **Ingestion modules**: `ingestion/db_ops/`, `ingestion/embeddings/`, `ingestion/pdf_extract.py`, `ingestion/chunking.py`
- **LLM wrapper**: `inference/llm/wrapper.py`
- **Utility functions**: Helper functions, logger functions, wrapper functions

### Integration Tests

Integration tests cover:
- **LLM Provider Integration**: Dynamic connectivity tests for OpenAI, Google Gemini, Ollama based on `.env` variables
  - Tests automatically skip providers that are not configured
  - Verifies API connectivity and response format
- **Database Schema Verification**: Comprehensive tests for database initialization (`test_database_schema.py`)
  - Verifies all three required tables exist: `documents`, `chunks`, `thread_tracking`
  - Validates table structure (columns, data types, constraints)
  - Checks required indexes exist for efficient retrieval
  - Verifies multi-modal embedding support (vector type, dimensions)
  - Tests run automatically after `make up` to ensure schema is properly initialized
- **End-to-End Pipeline**: Full ingestion → retrieval → synthesis workflows
- **Database Operations**: Real database interactions with test fixtures

## LLM Provider Integration Tests

Integration tests dynamically check connectivity with LLM providers based on `.env` variables:
- **OpenAI**: Tests if `OPENAI_API_KEY` is set
- **Google Gemini**: Tests if `GEMINI_API_KEY` is set
- **Ollama**: Tests if `OLLAMA_URL` is set

Tests automatically skip providers that are not configured, so you don't need to manually execute per provider.

## Test Configuration

Test configuration is defined in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

## Installing Test Dependencies

Install test dependencies:
```bash
pip install -e ".[dev]"
```

This installs:
- `pytest>=7.0.0` - Test framework
- `pytest-cov>=4.0.0` - Coverage reporting

---

# 📊 Agentic Reasoning Logs

## Overview

Deep RAG includes comprehensive logging of all agentic reasoning steps for:
- **Future Model Training**: CSV format for supervised fine-tuning (SFT) datasets
- **Presentation Materials**: Human-readable TXT format for demonstrations
- **Debugging & Analysis**: Detailed trace of retrieval and reasoning decisions

## Log Files

Logs are automatically saved to `inference/graph/logs/` with timestamps:

```
inference/graph/logs/
├── agent_log_20250106_143052.csv  # Structured data for training
└── agent_log_20250106_143052.txt  # Human-readable for presentations
```

## What Gets Logged

### CSV Format (for Training)
- Timestamp
- Session ID (for tracking conversations)
- Node (planner, retriever, compressor, critic, synthesizer)
- Action (plan, retrieve, compress, evaluate, synthesize)
- Question & Plan
- Query used for retrieval
- Number of chunks retrieved
- Pages retrieved (as JSON array)
- Confidence score
- Iterations count
- Refinements (sub-queries)
- Final answer
- Metadata (scores, timings, etc.)

### TXT Format (for Presentations)
Human-readable format with:
- Timestamped steps
- Query and plan details
- Retrieval results with text previews
- Confidence evaluations
- Refinement decisions
- Final answer with sources

## Log Access

Logs are generated automatically during LangGraph pipeline execution (`/ask-graph`, `/infer-graph` endpoints). No configuration needed.

---

# 🗺️ Graph Visualization

Export the LangGraph pipeline diagram as a PNG (requires Graphviz). If Graphviz isn't installed, a Mermaid file is produced instead.

**Pipeline Flow:** `planner → retriever → compressor → critic → (refine_retrieve ↺ <= 3 limit) → synthesizer`

### Install Graphviz
```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz
```

### Export Graph
```bash
# Via CLI
python inference/cli.py graph --out deep_rag_graph.png

# Via Make
make graph OUT=deep_rag_graph.png DOCKER=true

# Via TOML
deep-rag graph --out deep_rag_graph.png

# Via REST API
curl "http://localhost:8000/graph?out=deep_rag_graph.png" -o deep_rag_graph.png
```

---

# 🗄️ Database Management

## Restart Database
```bash
docker compose restart db
```

## Rebuild Indexes
```bash
docker compose exec db psql -U $DB_USER -d $DB_NAME -c "REINDEX TABLE chunks;"
```

## Verify Schema
```bash
docker compose exec db psql -U $DB_USER -d $DB_NAME -c "\dt"
```

---

# 🔧 Troubleshooting

## Verify All Pages Were Ingested
```bash
# Use inspect command to check page distribution
python inference/cli.py inspect --title "Document Title"
# Or via REST API
curl "http://localhost:8000/diagnostics/document?doc_title=Your%20Title"
```

## Check Retrieval Logs
The system logs show which pages are represented in retrieved chunks:

`cat inference/graph/logs/agent_log_*.txt`
- Look for "Pages represented in retrieved chunks: [1, 2, ...]"
- Check text previews to see what content was actually retrieved

## Common Issues

**Issue**: Only page 1 content is being retrieved
- **Solution**: Check if all pages were ingested using `inspect` command
- **Solution**: Check retrieval logs for page distribution
- **Solution**: Verify chunking created chunks for all pages

**Issue**: Graph visualization fails
- **Solution**: Install Graphviz: `sudo apt-get install graphviz` or `brew install graphviz`
- **Solution**: System will fallback to Mermaid format if Graphviz is missing

---

# 📚 Additional Resources

- See `md_guides/` directory for detailed guides:
  - [`ENTRY_POINTS_AND_SCENARIOS.md`](md_guides/ENTRY_POINTS_AND_SCENARIOS.md) - Entry point scenarios and use cases
  - [`THREAD_TRACKING_AND_AUDIT.md`](md_guides/THREAD_TRACKING_AND_AUDIT.md) - Thread tracking and audit logging
  - [`LLM_SETUP.md`](md_guides/LLM_SETUP.md) - LLM provider configuration
  - [`EMBEDDING_OPTIONS.md`](md_guides/EMBEDDING_OPTIONS.md) - Embedding model options
  - [`SETUP_GUIDE.md`](md_guides/SETUP_GUIDE.md) - Detailed setup instructions
  - [`RESET_DB.md`](md_guides/RESET_DB.md) - Database reset procedures
