# Quick Start Guide

## Overview

Deep RAG is a multi-service application consisting of:
- **Database Service** (`vector_db`): PostgreSQL with pgvector extension
- **Backend API** (`deep_rag_backend`): FastAPI service for RAG operations
- **Frontend UI** (`deep_rag_frontend`): Streamlit web interface

## Prerequisites

1. **Python 3.11** (required for Google Gemini SDK compatibility)
   - Check version: `python --version` (should be 3.11 or higher)
   - Note: Python 3.10 support is deprecated in Google Gemini SDK as of 2026

2. **Docker & Docker Compose** (for containerized deployment)
   - Docker Desktop or Docker Engine
   - Docker Compose v2.0+

3. **Environment Configuration**
   - Create `.env` file in project root (see Environment Setup below)

## Project Structure

```
deep_rag/                          # Project root
├── deep_rag_backend/              # Backend API (FastAPI)
│   ├── inference/                 # API routes, agents, LangGraph
│   ├── ingestion/                 # Document ingestion pipeline
│   ├── retrieval/                 # Hybrid retrieval system
│   ├── tests/                     # Test suite
│   ├── Dockerfile                 # Backend Docker image (Python 3.11)
│   └── docker-compose.yml         # Backend standalone compose
│
├── deep_rag_frontend/             # Frontend UI (Streamlit)
│   ├── app.py                     # Main Streamlit app
│   ├── api_client.py              # API client wrapper
│   ├── Dockerfile                 # Frontend Docker image (Python 3.11)
│   └── docker-compose.yml         # Frontend standalone compose
│
├── vector_db/                     # Database schemas and migrations
│   ├── schema_multimodal.sql      # Main schema
│   └── migration_*.sql            # Migration scripts
│
├── docker-compose.yml             # Full stack orchestration
├── .env.example                   # Environment variables template
├── .gitignore                     # Root gitignore
└── README.md                      # Project documentation
```

## Environment Setup

### 1. Create Root `.env` File

From the project root directory (`deep_rag/`):

```bash
# Copy the example file
cp .env.example .env

# Edit .env and fill in your values
# Required variables:
#   - Database credentials (DB_USER, DB_PASS, DB_NAME)
#   - LLM API key (GEMINI_API_KEY)
#   - Embedding model (CLIP_MODEL, EMBEDDING_DIM)
```

### 2. Component-Specific `.env` Files (Optional)

If running services independently, you can create component-specific `.env` files:

**Backend:**
```bash
cd deep_rag_backend
cp .env.example .env
# Edit with backend-specific values
```

**Frontend:**
```bash
cd deep_rag_frontend
cp .env.example .env
# Edit with frontend-specific values (API_BASE_URL)
```

**Database:**
```bash
cd vector_db
cp .env.example .env
# Edit with database credentials
```

## Installation

### Option 1: Full Stack with Docker (Recommended)

Run all three services together (database, backend API, frontend):

**Using Make (from project root):**
```bash
# From project root (deep_rag/)
make up              # Start all services

# Services will be available at:
# - Frontend: http://localhost:8501
# - Backend API: http://localhost:8000
# - Database: localhost:5432
```

**Using Docker Compose directly:**
```bash
# From project root (deep_rag/)
docker-compose up -d

# Services will be available at:
# - Frontend: http://localhost:5173
# - Backend API: http://localhost:8000
# - Database: localhost:5432
```

**View logs:**
```bash
# Using Make (from project root)
make logs            # Tail logs from all services

# Using Docker Compose directly
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f frontend
docker-compose logs -f db
```

**Stop services:**
```bash
# Using Make (from project root)
make down            # Stop all services

# Using Docker Compose directly
docker-compose down
```

### Option 2: Independent Services

#### Backend Only

```bash
cd deep_rag_backend
docker-compose up -d

# Backend API: http://localhost:8000
# Note: Requires database to be running separately
```

#### Frontend Only

```bash
cd deep_rag_frontend
docker-compose up -d

# Frontend: http:/localhost:5173
# Note: Requires backend API to be running
# Set API_BASE_URL in .env to point to backend
```

#### Database Only

```bash
cd vector_db
docker-compose up -d

# Database: localhost:5432
# Note: Other services can connect to this database
```

### Option 3: Local Development

#### Backend (Local)

```bash
cd deep_rag_backend

# Install dependencies
pip install -r requirements.txt

# Set environment variables
# Copy .env.example to .env and fill in values

# Run the API
python -m inference.service

# Or use uvicorn directly
uvicorn inference.service:app --reload
```

#### Frontend (Local)

```bash
cd deep_rag_frontend

# Install dependencies
pip install -r requirements.txt

# Set environment variables
# Copy .env.example to .env and set API_BASE_URL

# Run Streamlit
streamlit run app.py
```

**Note:** Local development requires:
- Python 3.11 or higher
- Database running (Docker or local PostgreSQL)
- Backend API running (for frontend)

## First Steps

### 1. Verify Services Are Running

**Check Backend API:**
```bash
curl http://localhost:8000/health
```

**Check Frontend:**
```bash
curl http://localhost:5173/health
```

**Check Database:**
```bash
# From Docker container
docker-compose exec db psql -U $DB_USER -d $DB_NAME -c "SELECT version();"
```

### 2. Access the Frontend

Open your browser and navigate to:
```
http://localhost:5173/
```

### 3. Upload a Document

1. Use the sidebar file uploader to ingest a PDF, TXT, or image
2. Or attach a file to your question to ingest and query simultaneously

### 4. Ask Questions

1. Type your question in the chat input
2. Press Enter to send
3. View the response and metadata

## Features

### Chat Interface
- Type questions in the chat input
- View conversation history
- See response metadata
- Thread management (create new threads)

### File Upload
- **Single Upload**: Upload files in the sidebar
- **Batch Upload**: Select multiple files at once
- **Attach to Message**: Upload files with your question

### Document Management
- **View Documents**: See all ingested documents in sidebar
- **Filter Queries**: Query specific documents
- **Cross-Document Search**: Enable in settings

### Thread Management
- **New Thread**: Start a fresh conversation
- **Thread History**: (Coming soon) View previous threads
- Each thread maintains its own conversation context

## Troubleshooting

### Python Version Issues

**Error: "Python 3.11 required"**
- Ensure Python 3.11 or higher is installed
- Check version: `python --version`
- Docker images use Python 3.11 automatically

**Error: "Google Gemini SDK compatibility"**
- Google Gemini SDK requires Python 3.11+
- Python 3.10 support is deprecated as of 2026

### API Connection Issues

**Frontend can't connect to backend:**
- Ensure backend is running: `curl http://localhost:8000/health`
- Check `API_BASE_URL` in frontend `.env`:
  - Full stack: `http://api:8000`
  - Standalone with local backend: `http://host.docker.internal:8000`
  - Standalone with remote backend: `http://your-backend-url:8000`

**Backend can't connect to database:**
- Ensure database is running
- Check database credentials in `.env`
- Verify `DB_HOST` is correct:
  - Full stack: `db`
  - Standalone: `localhost` or database hostname

### File Upload Issues

- Supported formats: PDF, TXT, PNG, JPEG
- Check file size limits
- Ensure backend has sufficient resources
- Verify backend API is accessible

### Docker Issues

**Container won't start:**
```bash
# Check logs
docker-compose logs [service_name]

# Check container status
docker-compose ps
```

**Port already in use:**
- Change port in `.env` or `docker-compose.yml`
- Or stop the conflicting service

**Build errors:**
- Ensure Docker has sufficient resources
- Check Dockerfile syntax
- Verify all dependencies are available

### Import Errors

**Python dependencies:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.11+
```

**Module not found:**
- Verify `PYTHONPATH` is set correctly
- Check that you're in the correct directory
- Ensure virtual environment is activated (if using one)

## Service-Specific Guides

- **Backend**: See `deep_rag_backend/README.md` or `deep_rag_backend/md_guides/`
- **Frontend**: See `deep_rag_frontend/README.md`
- **Database**: See `vector_db/` for schema and migration files

## Next Steps

- Review `README.md` for detailed documentation
- Check `deep_rag_frontend/SUGGESTED_ROUTES.md` for recommended backend enhancements
- See `deep_rag_frontend/LANGGRAPH_CONSIDERATIONS.md` for thread management details
- Explore `deep_rag_backend/md_guides/` for backend-specific guides
