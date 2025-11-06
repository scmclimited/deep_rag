# Setup Guide: Deep RAG

## Environment Configuration

### For Docker Deployment (Recommended)

When running in Docker, the database hostname should be `db` (the Docker service name).

Create a `.env` file in the project root:

```bash
# Database Configuration (for Docker)
DB_HOST=db
DB_PORT=5432
DB_USER=rag
DB_PASS=rag
DB_NAME=ragdb

# LLM Provider Configuration
LLM_PROVIDER=llava
LLAVA_URL=http://localhost:11434
LLAVA_MODEL=llava-hf/llava-1.5-7b-hf
LLM_TEMPERATURE=0.2
```

### For Local Development

When running locally (outside Docker), use `localhost`:

```bash
# Database Configuration (for local)
DB_HOST=localhost
DB_PORT=5432
DB_USER=rag
DB_PASS=rag
DB_NAME=ragdb

# LLM Provider Configuration
LLM_PROVIDER=llava
LLAVA_URL=http://localhost:11434
LLAVA_MODEL=llava-hf/llava-1.5-7b-hf
LLM_TEMPERATURE=0.2
```

## Common Issues

### Issue 1: "could not translate host name" Error

**Symptom**: `OperationalError: could not translate host name "Locahost" to address`

**Cause**: Typo in `DB_HOST` environment variable (e.g., "locahost" instead of "localhost")

**Solution**: 
- Check your `.env` file for typos
- Ensure `DB_HOST=db` when running in Docker
- Ensure `DB_HOST=localhost` when running locally

### Issue 2: Title Not Provided

**Symptom**: Title is None when ingesting PDFs

**Solution**: The code now automatically extracts titles from:
1. Provided title parameter (if given)
2. PDF metadata title
3. First line of first page
4. Filename (as fallback)

You can also explicitly provide a title:
```bash
docker compose exec api python -m inference.cli ingest "file.pdf" --title "My Document Title"
```

## Quick Start

### 1. Start Docker Services
```bash
docker compose up -d --build
```

### 2. Verify Services Are Running
```bash
docker ps
# Should show: deep_rag_pgvector and deep_rag_api
```

### 3. Ingest a PDF
```bash
# Inside Docker (recommended)
docker compose exec api python -m inference.cli ingest "NYMBL - AI Engineer - Omar.pdf"

# Or with makefile
make cli-ingest FILE="NYMBL - AI Engineer - Omar.pdf" DOCKER=true
```

### 4. Query the Document
```bash
docker compose exec api python -m inference.cli query "What are the requirements?"
```

## Troubleshooting

### Database Connection Issues

1. **Verify database is running**:
   ```bash
   docker compose ps
   docker compose logs db
   ```

2. **Check environment variables**:
   ```bash
   docker compose exec api env | grep DB_
   ```

3. **Test database connection**:
   ```bash
   docker compose exec db psql -U rag -d ragdb -c "SELECT 1;"
   ```

### Import Errors

If you see module import errors, ensure:
- Docker container was rebuilt: `docker compose up -d --build`
- All dependencies are installed: `pip install -r requirements.txt` (for local runs)

