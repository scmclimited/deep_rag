# 🧠 Deep RAG — Agentic Retrieval-Augmented Generation Pipeline

Deep RAG is a modular, production-ready Retrieval-Augmented Generation (RAG) system for querying and reasoning over PDFs (text + images).  
It combines deterministic PDF parsing with hybrid (lexical + vector) retrieval, cross-encoder reranking, and an agentic multi-stage reasoning loop inspired by “deep-thinking RAG” architectures.

---

# 📂 Directory Structure
```bash
deep_rag/
├── inference/
│   ├── agent_loop.py          # Direct pipeline: supervisor-style reasoning loop (plan → retrieve → compress → reflect → synthesize)
│   ├── cli.py                 # Typer CLI interface matching FastAPI service routes
│   ├── llm.py                 # Centralized LLM provider interface (LLaVA, OpenAI, Ollama, Gemini)
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
│   └── embeddings.py         # Multi-modal embedding system (CLIP) for unified text/image embeddings
│
├── retrieval/
│   └── retrieval.py           # Hybrid retrieval: lexical (pg_trgm) + vector (pgvector) + cross-encoder reranking
│
├── vector_db/
│   ├── schema_multimodal.sql  # Multi-modal schema for CLIP embeddings (512 dims) with content_type/image_path
│   ├── migration_add_multimodal.sql  # Migration script for adding multi-modal support to existing DB
│   ├── ingestion_schema.sql   # Legacy schema (for reference)
│   └── docker-compose.yml     # Stand-alone DB service (pgvector)
│
├── md_guides/                 # Documentation guides
│   ├── EMBEDDING_OPTIONS.md   # Embedding model options and recommendations
│   ├── LLM_SETUP.md           # LLM provider setup guide (OpenAI, Ollama, LLaVA, Gemini)
│   ├── PROJECT_ASSESSMENT.md  # Project assessment requirements
│   ├── RESET_DB.md            # Database reset instructions
│   └── SETUP_GUIDE.md         # Detailed setup guide
│
├── .env                       # Environment variables (database credentials, LLM provider settings)
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project metadata and dependencies
├── makefile                  # Convenience make commands for common tasks
├── Dockerfile                 # API container definition
└── docker-compose.yml         # Root-level orchestration (API + DB)

```


---

# 🚀 Features

| Layer | Description |
|-------|--------------|
| **Ingestion** | Extracts text and captions from PDFs using PyMuPDF (`fitz`) with OCR fallback (`pytesseract`). Chunks and embeds with `BAAI/bge-m3` into Postgres + pgvector. |
| **Retrieval** | Hybrid search combining pg_trgm (BM25-style) lexical scores + vector similarity. Reranked by a cross-encoder (`bge-reranker-base`). |
| **Agentic Loop** | Supervisor graph: plan → retrieve → compress → reflect → synthesize. Reflection loop auto-refines queries when confidence < threshold. |
| **Microservice Ready** | FastAPI REST interface for programmatic queries (`/ask`) and health checks (`/health`). |
| **CLI Ready** | Typer CLI for ingestion and question answering. |
| **Containerized DB** | pgvector/pg16 Docker image with automatic schema init via mounted SQL file. |

---

# 🎯 Entry Point Mapping

The Deep RAG system provides multiple entry points for different use cases:

## CLI Commands ↔ Make Scripts ↔ REST Endpoints

| CLI Command | Make Script | REST Endpoint | Pipeline | Purpose |
|------------|-------------|---------------|----------|---------|
| `ingest` | `make cli-ingest` | `POST /ingest` | Direct | **Ingestion only**: Embeds documents into vector DB without querying. Use when you want to pre-populate your knowledge base. |
| `query` | `make query` | `POST /ask` | Direct (`agent_loop.py`) | **Query only (direct pipeline)**: Fast, deterministic pipeline for simple queries. No conditional routing. Best for straightforward questions. |
| `query-graph` | `make query-graph` | `POST /ask-graph` | LangGraph | **Query only (LangGraph)**: Agentic pipeline with conditional routing. Agents can refine queries and retrieve more evidence if confidence is low. Best for complex questions requiring iterative reasoning. |
| `infer` | N/A (use CLI) | `POST /infer` | Direct (`agent_loop.py`) | **Ingest + Query (direct pipeline)**: Combined ingestion and querying in one operation. Use when you have a document and want immediate answers with fast, deterministic processing. Use CLI: `python inference/cli.py infer "question" --file path/to/file.pdf` |
| `infer-graph` | `make infer-graph` | `POST /infer-graph` | LangGraph | **Ingest + Query (LangGraph)**: Combined ingestion with agentic reasoning. Best when you need to ingest and then perform complex reasoning over the new content. Supports file upload, title, and thread_id. |
| `health` | N/A | `GET /health` | N/A | **Health check**: Verifies database connectivity and service availability. |
| `graph` | `make graph` | `GET /graph` | N/A | **Graph export**: Exports LangGraph pipeline visualization as PNG or Mermaid diagram. Useful for understanding the agentic flow. |

### Pipeline Comparison

- **Direct Pipeline** (`agent_loop.py`): Linear execution, faster, deterministic. Best for simple queries.
- **LangGraph Pipeline** (`graph/graph_wrapper.py`): Conditional routing, agents can refine queries, iterative retrieval. Best for complex questions requiring multi-step reasoning.

## REST API Examples

### POST /ask (Direct Pipeline)
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the AI solutions engineer technical assessment about?"}'
```

### POST /ask-graph (LangGraph Pipeline)
```bash
curl -X POST http://localhost:8000/ask-graph \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the specific requirements?", "thread_id": "session-1"}'
```

### POST /ingest (File Upload)
```bash
curl -X POST http://localhost:8000/ingest \
  -F "attachment=@path/to/file.pdf" \
  -F "title=Optional Document Title"
```

### POST /infer (Ingest + Query - Direct)
```bash
curl -X POST http://localhost:8000/infer \
  -F "question=What does this document say about RAG systems?" \
  -F "attachment=@path/to/file.pdf" \
  -F "title=Optional Title"
```

### POST /infer-graph (Ingest + Query - LangGraph)
```bash
curl -X POST http://localhost:8000/infer-graph \
  -F "question=What are the key requirements for this RAG system?" \
  -F "attachment=@path/to/file.pdf" \
  -F "title=Optional Title" \
  -F "thread_id=session-1"
```

### GET /health
```bash
curl http://localhost:8000/health
```

### GET /graph
```bash
curl "http://localhost:8000/graph?out=deep_rag_graph.png"
```

# 🧩 Prerequisites

- Python ≥ 3.11 (required for Google Gemini support due to 3.10 support deprecation in 2026)  
- Docker & Docker Compose  
- (Optional) Tesseract OCR + Poppler (for scanned PDFs)

## Install system dependencies
```bash
Ubuntu/Debian: sudo apt install tesseract-ocr poppler-utils -y
# or
macOS: brew install tesseract poppler
```

## Setup (Local Development)
### Clone the repository
```bash
git clone https://github.com/omoral02/deep_rag.git
cd deep_rag
```

### Create and activate a virtual environment
```bash 
python -m venv .venv
source .venv/bin/activate      # On Windows use: .venv\Scripts\activate
```

#### Upgrade pip and install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Create a `.env` file in the project root:
```bash
DB_HOST=localhost
DB_PORT=5432
DB_USER=user_here
DB_PASS=password_here
DB_NAME=ragdb
```

## Usage
### Option 1 Vector DB initialization
```bash
cd vector_db
docker-compose up -d --build
```

### Option 2 from project root
```bash
docker compose up -d --build
```

### Verify
```bash
docker ps
```
- deep_rag_pgvector and deep_rag_api should be running

## CLI Usage

### Option 1: Run Inside Docker (Recommended)
All dependencies are pre-installed in the Docker container.

```bash
# Start the stack
docker compose up -d --build

# Ingest a PDF (inside Docker)
# Note: PDFs should be in inference/samples/ directory or provide full path relative to /app
# Title will be automatically extracted from PDF metadata or first page
docker compose exec api python -m inference.cli ingest "inference/samples/NYMBL - AI Engineer - Omar.pdf"

# Or provide a custom title explicitly
docker compose exec api python -m inference.cli ingest "inference/samples/NYMBL - AI Engineer - Omar.pdf" --title "AI Engineer Assessment"

# Or using makefile (automatic title extraction)
make cli-ingest FILE="inference/samples/NYMBL - AI Engineer - Omar.pdf" DOCKER=true

# Or with custom title using makefile
make cli-ingest FILE="inference/samples/NYMBL - AI Engineer - Omar.pdf" DOCKER=true TITLE="AI Engineer Assessment"

# Or with custom title (direct docker command)
docker compose exec api python -m inference.cli ingest "inference/samples/NYMBL - AI Engineer - Omar.pdf" --title "AI Engineer Assessment"

# Ask a question (inside Docker)
docker compose exec api python -m inference.cli query "What are the requirements?"
```

### Option 2: Run Locally
Install dependencies first, then run locally.

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest a PDF (locally)
# Note: PDFs should be in inference/samples/ directory or provide full path
# Title will be automatically extracted from PDF metadata or first page
python inference/cli.py ingest "inference/samples/NYMBL - AI Engineer - Omar.pdf"

# Or provide a custom title explicitly
python inference/cli.py ingest "inference/samples/NYMBL - AI Engineer - Omar.pdf" --title "AI Engineer Assessment"

# Ask a question (locally)
python inference/cli.py query "What are the main sections discussed in the document?"
```

### Ask with Lang Graph
```bash
# After ingesting one or more PDFs
python inference/run_graph.py "Summarize the methodology"
# Or
docker compose up -d --build
curl -s -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main sections?","thread_id":"run1"}'
```

---
## With TOML
```bash 
deep-rag ingest path/to/file.pdf
deep-rag query "Explain the methodology section"
deep-rag graph --out deep_rag_graph.png

```
---

## Health Check
```bash
curl -s localhost:8000/health`
```

## Ask a question
```bash
curl -s -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the results section"}'
```

## Quickstart Scripts
### 1. Launch DB
```bash
cd vector_db
docker-compose up -d
```

### 2. (Optional) Verify schema
```bash 
psql -h localhost -U rag -d ragdb -c '\dt'`
```

### 3. Ingest a document
```bash
python ingestion/ingest.py ./samples/my_pdf.pdf
```

### 4. Query the document
```bash
python inference/agent_loop.py "What is discussed in the pdf?"
```

### 1.2. Full microservice run from project root
```bash
cd deep_rag
docker compose up -d --build
```

### 1.3. cURL Test
```bash
`curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"Summarize the PDF"}'
```
---

# DB Management
- Restart DB
```bash
docker compose restart db
```

- Rebuild Indexes
```bash
REINDEX TABLE chunks;
```

---
# 🗺️ Graph Visualization

Export the LangGraph pipeline diagram as a PNG (requires Graphviz). If Graphviz
isn’t installed, a Mermaid file is produced instead.

`planner → retriever → compressor → critic → (refine_retrieve ↺) → synthesizer`

### Install Graphviz to enable PNG export:
```bash
Ubuntu/Debian: sudo apt-get install graphviz
# or 
macOS: brew install graphviz
```

```bash
# PNG (Graphviz) or Mermaid fallback
python inference/graph_viz.py --out deep_rag_graph.png

# If using the CLI wrapper:
python cli.py graph --out deep_rag_graph.png

# or through project scripts
deep-rag graph --out deep_rag_graph.png
```

# Make scripts (should you prefer this route)

## Install Make on local hardware if not available/found using this command
`make --version`

### Per OS:
Debian/Ubuntu/WSL:
```bash 
sudo apt update
sudo apt install -y make build-essential
```
macOS: `xcode-select --install`

Windows: `choco install make`

```bash
# Start full stack (API + DB)
make up

# Turn down services
make down

# Ingest a PDF (title will be automatically extracted from PDF metadata or first page)
# Note: When using DOCKER=true, paths are relative to /app (project root)
make ingest FILE=inference/samples/NYMBL\ -\ AI\ Engineer\ -\ Omar.pdf

# Or ingest using CLI with automatic title extraction
make cli-ingest FILE="inference/samples/NYMBL - AI Engineer - Omar.pdf" DOCKER=true

# Or ingest with a custom title using makefile
make cli-ingest FILE="inference/samples/NYMBL - AI Engineer - Omar.pdf" DOCKER=true TITLE="My Custom Document Title"

# Or with custom title (direct docker command)
docker compose exec api python -m inference.cli ingest "./samples/my_pdf.pdf" --title "My Custom Document Title"

# Ask a question (runs the inferene/agent_loop.py pipeline)
make query Q="Summarize the methodology section"

# Query with LangGraph (runs the inference/graph/graph_wrapper.py agentic deep reasoning pipeline)
make query-graph Q='your question here'

# Ingest + Query with LangGraph (runs the inference/graph/graph_wrapper.py agentic reasoning pipeline)
make infer-graph Q='your question' FILE=path/to/file.pdf TITLE='Document Title'

# Export the LangGraph diagram (PNG; Mermaid if Graphviz isn't installed)
make graph DOCKER=true OUT=artifacts/deep_rag_graph.png

```
