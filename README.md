# 🧠 Deep RAG — Agentic Retrieval-Augmented Generation Pipeline

Deep RAG is a modular, production-ready Retrieval-Augmented Generation (RAG) system for querying and reasoning over PDFs (text + images).  
It combines deterministic PDF parsing with hybrid (lexical + vector) retrieval, cross-encoder reranking, and an agentic multi-stage reasoning loop inspired by "deep-thinking RAG" architectures.

---

# 📂 Directory Structure
```bash
deep_rag/
├── inference/
│   ├── agent_loop.py          # Direct pipeline: supervisor-style reasoning loop (plan → retrieve → compress → reflect → synthesize)
│   ├── cli.py                 # Typer CLI interface matching FastAPI service routes
│   ├── llm.py                 # Centralized LLM provider interface (currently using Gemini)
│   ├── service.py             # FastAPI REST API entrypoint with all endpoints
│   ├── graph/
│   │   ├── graph.py           # LangGraph pipeline definition with conditional routing nodes
│   │   ├── graph_wrapper.py   # Wrapper for LangGraph pipeline with logging
│   │   └── graph_viz.py       # Graph visualization export (PNG/Mermaid)
│   └── samples/               # Sample PDFs for testing
│
├── ingestion/
│   ├── ingest.py              # PDF ingestion: deterministic PDF→text/OCR pipeline with embedding + pgvector upsert
│   ├── ingest_text.py         # Plain text file ingestion
│   ├── ingest_image.py        # Image file ingestion (PNG, JPEG) with OCR extraction
│   └── embeddings.py          # Multi-modal embedding system (CLIP) for unified text/image embeddings
│
├── retrieval/
│   ├── retrieval.py           # Hybrid retrieval: lexical (pg_trgm) + vector (pgvector) + cross-encoder reranking
│   └── diagnostics.py         # Diagnostic tools for inspecting stored chunks and pages
│
├── vector_db/
│   ├── schema_multimodal.sql  # Multi-modal schema for CLIP embeddings (512 dims) with content_type/image_path
│   ├── migration_add_multimodal.sql  # Migration script for adding multi-modal support to existing DB
│   ├── ingestion_schema.sql   # Legacy schema (for reference)
│   └── docker-compose.yml     # Stand-alone DB service (pgvector)
│
├── md_guides/                 # Documentation guides
│   ├── EMBEDDING_OPTIONS.md   # Embedding model options and recommendations
│   ├── LLM_SETUP.md           # LLM provider setup guide
│   ├── PROJECT_ASSESSMENT.md  # Project assessment requirements
│   ├── RESET_DB.md            # Database reset instructions
│   └── SETUP_GUIDE.md         # Detailed setup guide
│
├── .env                       # Environment variables (database credentials, LLM provider settings)
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata and dependencies
├── makefile                   # Convenience make commands for common tasks
├── Dockerfile                 # API container definition
└── docker-compose.yml         # Root-level orchestration (API + DB)

```

---

# 🚀 Features

| Layer | Description |
|-------|--------------|
| **Ingestion** | Multi-modal ingestion: PDFs (text + OCR), plain text files, and images (PNG/JPEG). Extracts text using PyMuPDF (`fitz`) with OCR fallback (`pytesseract`). Chunks and embeds with **CLIP-ViT-L/14** (`sentence-transformers/clip-ViT-L-14`) into unified **768-dimensional vectors** in Postgres + pgvector. **Upgraded from ViT-B/32 (512 dims)** for better semantic representation. |
| **Retrieval** | Hybrid search combining pg_trgm (BM25-style) lexical scores + vector similarity (cosine distance). Reranked by a cross-encoder (`bge-reranker-base`). Supports multi-modal queries (text + images). Dynamically supports 512 or 768 dimensional embeddings. |
| **Agentic Loop** | Two pipeline options: (1) **Direct pipeline** (`agent_loop.py`): Linear execution (plan → retrieve → compress → reflect → synthesize). (2) **LangGraph pipeline** (`graph/graph_wrapper.py`): Conditional routing with iterative refinement - agents can refine queries and retrieve more evidence when confidence < threshold. **Includes comprehensive logging** to CSV/TXT for future SFT training. |
| **LLM Integration** | Currently using **Google Gemini** for agentic reasoning. **Recommended models**: `gemini-1.5-flash` (1M token context, best balance) or `gemini-2.0-flash` (latest, improved reasoning). Alternative: `gemini-2.5-flash-lite` for faster, lightweight processing. Code structure supports future integration with OpenAI, Ollama, and LLaVA. |
| **Multi-modal Support** | Unified embedding space for text and images using CLIP-ViT-L/14 (768 dims), enabling semantic search across different content types with better representation than ViT-B/32. |
| **Reasoning Logs** | **NEW**: All agentic reasoning steps are logged to `inference/graph/logs/` in both CSV (for training) and TXT (for presentations). Captures queries, plans, retrievals, confidence scores, and refinements for future SFT model training. |
| **Microservice Ready** | FastAPI REST interface with comprehensive endpoints for ingestion, querying (direct and LangGraph), and health checks. |
| **CLI Ready** | Typer CLI matching all REST endpoints for easy local development and testing. |
| **Containerized DB** | pgvector/pg16 Docker image with automatic schema init via mounted SQL file. Support for fresh database starts and migrations. |

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

### Token Limit Handling

CLIP models have a 77 token limit (inherent to architecture). Deep RAG handles this through:
- **Intelligent Chunking**: Chunks are limited to ~25 words (≈32-37 tokens) with safe margin
- **Token-Aware Truncation**: Uses CLIP's tokenizer for accurate truncation
- **Fallback Handling**: Conservative word-based truncation if tokenizer unavailable
- **Retry Logic**: Multiple truncation attempts with progressively smaller chunks

### Upgrading from ViT-B/32 to ViT-L/14

If you have an existing database with ViT-B/32 (512 dims), see `vector_db/migration_upgrade_to_768.sql` for migration steps.

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

## Use Cases

### 1. Future SFT Training
```python
import pandas as pd

# Load reasoning logs for training
df = pd.read_csv("inference/graph/logs/agent_log_20250106_143052.csv")

# Filter successful queries with high confidence
training_data = df[df['confidence'] > 0.7]

# Extract query-answer pairs
for _, row in training_data.iterrows():
    question = row['question']
    plan = row['plan']
    answer = row['answer']
    # Use for fine-tuning your own model
```

### 2. Presentation & Demos
```bash
# View human-readable logs
cat inference/graph/logs/agent_log_20250106_143052.txt

# Show reasoning trace with timestamps
grep "GRAPH NODE" inference/graph/logs/*.txt
```

### 3. Performance Analysis
- Identify which queries require refinements
- Track retrieval quality (pages found, confidence scores)
- Analyze chunking effectiveness
- Optimize retrieval parameters

## Log Access

Logs are generated automatically during LangGraph pipeline execution (`/ask-graph`, `/infer-graph` endpoints). No configuration needed.

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

### Why Gemini for RAG?

1. **Large Context Window**: 1M tokens handles large documents and many chunks
2. **Fast Response Times**: Critical for interactive Q&A
3. **Strong Instruction Following**: Important for structured prompts (planning, compression)
4. **Cost Effective**: Better $/token ratio than competitors
5. **Multi-Turn Support**: Good for refinement loops in agentic pipeline

---

# 🎯 Entry Point Mapping

The Deep RAG system provides multiple entry points for different use cases:

## CLI Commands ↔ Make Scripts ↔ REST Endpoints

| CLI Command | Make Script | REST Endpoint | Pipeline | Purpose |
|------------|-------------|---------------|----------|---------|
| `ingest` | `make cli-ingest` | `POST /ingest` | Direct | **Ingestion only**: Embeds documents into vector DB without querying. Use when you want to pre-populate your knowledge base. |
| `query` | `make query` | `POST /ask` | Direct (`agent_loop.py`) | **Query only (direct pipeline)**: Fast, deterministic pipeline for simple queries. No conditional routing. Best for straightforward questions. |
| `query-graph` | `make query-graph` | `POST /ask-graph` | LangGraph | **Query only (LangGraph)**: Agentic pipeline with conditional routing. Agents can refine queries and retrieve more evidence if confidence is low. Best for complex questions requiring iterative reasoning. |
| `infer` | N/A (use CLI) | `POST /infer` | Direct (`agent_loop.py`) | **Ingest + Query (direct pipeline)**: Combined ingestion and querying in one operation. Use when you have a document and want immediate answers with fast, deterministic processing. |
| `infer-graph` | `make infer-graph` | `POST /infer-graph` | LangGraph | **Ingest + Query (LangGraph)**: Combined ingestion with agentic reasoning. Best when you need to ingest and then perform complex reasoning over the new content. Supports file upload, title, and thread_id. |
| `health` | N/A | `GET /health` | N/A | **Health check**: Verifies database connectivity and service availability. |
| `graph` | `make graph` | `GET /graph` | N/A | **Graph export**: Exports LangGraph pipeline visualization as PNG or Mermaid diagram. Useful for understanding the agentic flow. |
| `inspect` | `make inspect` | `GET /diagnostics/document` | N/A | **Document diagnostics**: Inspects what chunks and pages are stored for a document. Shows page distribution, chunk counts, and sample text. Essential for debugging ingestion and retrieval issues. |

### Pipeline Comparison

- **Direct Pipeline** (`agent_loop.py`): Linear execution, faster, deterministic. Best for simple queries.
- **LangGraph Pipeline** (`graph/graph_wrapper.py`): Conditional routing, agents can refine queries, iterative retrieval. Best for complex questions requiring multi-step reasoning.

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
Create a `.env` file in the project root:
```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=user_here
DB_PASS=password_here
DB_NAME=ragdb

# LLM Configuration (Currently using Gemini)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
# Recommended models (in order of preference):
# - gemini-1.5-flash: Best balance of speed and quality, 1M token context
# - gemini-2.0-flash: Latest model with improved reasoning
# - gemini-2.5-flash-lite: Lightweight, faster but limited context
GEMINI_MODEL=gemini-1.5-flash
LLM_TEMPERATURE=0.2

# Embedding Configuration (Multi-modal)
# CLIP-ViT-L/14: 768 dimensions (recommended, better quality)
# CLIP-ViT-B/32: 512 dimensions (faster, legacy)
CLIP_MODEL=sentence-transformers/clip-ViT-L-14
EMBEDDING_DIM=768
```

### 5. Start Services
```bash
# Option 1: Start full stack (API + DB)
docker compose up -d --build

# Option 2: Start DB only (for local development)
cd vector_db
docker-compose up -d
```

### 6. Verify Services
```bash
docker ps
# Should show: deep_rag_pgvector and deep_rag_api running
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
python inference/cli.py query-graph "What are the requirements?" --thread-id session-1

# Ingest + Query in one command
python inference/cli.py infer "What does this document say?" --file "path/to/file.pdf"
python inference/cli.py infer-graph "Analyze this document" --file "path/to/file.pdf" --title "Doc Title" --thread-id session-1

# Inspect stored chunks and pages (debugging)
python inference/cli.py inspect --title "Document Title"
python inference/cli.py inspect --doc-id your-doc-id-here
python inference/cli.py inspect  # List all documents

# Export graph visualization
python inference/cli.py graph --out deep_rag_graph.png

# Health check
python inference/cli.py health
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
docker compose exec api python -m inference.cli query-graph "Complex question" --thread-id session-1

# Inspect stored chunks (inside Docker)
docker compose exec api python -m inference.cli inspect --title "Document Title"
docker compose exec api python -m inference.cli inspect --doc-id your-doc-id-here
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
make query Q="Your question here" DOCKER=true
make query-graph Q="Complex question" DOCKER=true THREAD_ID=session-1

# Ingest + Query
make infer-graph Q="Your question" FILE="path/to/file.pdf" TITLE="Title" DOCKER=true THREAD_ID=session-1

# Diagnostics
make inspect TITLE="Document Title" DOCKER=true
make inspect DOC_ID=your-doc-id-here DOCKER=true
make inspect DOCKER=true  # List all documents

# Graph visualization
make graph OUT=deep_rag_graph.png DOCKER=true
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
deep-rag query-graph "What are the requirements?" --thread-id session-1

# Ingest + Query
deep-rag infer "Question" --file path/to/file.pdf
deep-rag infer-graph "Question" --file path/to/file.pdf --title "Title" --thread-id session-1

# Diagnostics
deep-rag inspect --title "Document Title"
deep-rag inspect --doc-id your-doc-id-here
deep-rag inspect  # List all documents

# Graph visualization
deep-rag graph --out deep_rag_graph.png
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
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the document about?"}'
```

#### POST /ask-graph (Query - LangGraph Pipeline)
```bash
curl -X POST http://localhost:8000/ask-graph \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the specific requirements?", "thread_id": "session-1"}'
```

#### POST /infer (Ingest + Query - Direct)
```bash
curl -X POST http://localhost:8000/infer \
  -F "question=What does this document say about RAG systems?" \
  -F "attachment=@path/to/file.pdf" \
  -F "title=Optional Title"
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
  - `LLM_SETUP.md` - LLM provider configuration
  - `EMBEDDING_OPTIONS.md` - Embedding model options
  - `SETUP_GUIDE.md` - Detailed setup instructions
  - `RESET_DB.md` - Database reset procedures
