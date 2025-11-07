# Environment Configuration Guide

This guide explains all environment variables used in Deep RAG.

## Core Configuration (.env)

Create a `.env` file in the project root with the following variables:

```bash
# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# PostgreSQL with pgvector
DB_HOST=localhost
DB_PORT=5432
DB_USER=user_here
DB_PASS=password_here
DB_NAME=ragdb

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# LLM Provider (currently supports: gemini, openai, ollama)
LLM_PROVIDER=gemini

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Gemini Model Selection (Recommended models in order of preference):
# 1. gemini-1.5-flash (RECOMMENDED)
#    - Context: 1M tokens
#    - Speed: Fast
#    - Quality: Excellent reasoning
#    - Use: Production RAG with large documents
#
# 2. gemini-2.0-flash (LATEST)
#    - Context: 1M tokens
#    - Speed: Fast with improved performance
#    - Quality: Latest improvements
#    - Use: Production with latest capabilities
#
# 3. gemini-2.5-flash-lite (LIGHTWEIGHT)
#    - Context: Limited
#    - Speed: Very fast
#    - Quality: Good for simple queries
#    - Use: Development, cost-constrained environments
#    - Note: May struggle with complex multi-step reasoning

GEMINI_MODEL=gemini-1.5-flash

# LLM Temperature (0.0 = deterministic, 1.0 = creative)
LLM_TEMPERATURE=0.2

# =============================================================================
# EMBEDDING CONFIGURATION (Multi-Modal)
# =============================================================================

# CLIP Model Selection
# 
# Option 1: CLIP-ViT-L/14 (RECOMMENDED for Production)
#   - Dimensions: 768
#   - Quality: Better semantic representation
#   - Model Size: ~400MB
#   - Use Case: Production, high-quality retrieval
#
# Option 2: CLIP-ViT-B/32 (Legacy, Faster)
#   - Dimensions: 512
#   - Quality: Good, faster inference
#   - Model Size: ~150MB
#   - Use Case: Development, resource-constrained environments

CLIP_MODEL=sentence-transformers/clip-ViT-L-14
EMBEDDING_DIM=768

# For legacy/faster option:
# CLIP_MODEL=sentence-transformers/clip-ViT-B-32
# EMBEDDING_DIM=512

# =============================================================================
# OPTIONAL CONFIGURATION
# =============================================================================

# Project root (for path resolution in Docker)
PROJECT_ROOT=/app

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Agent log directory
AGENT_LOG_DIR=inference/graph/logs
```

## Configuration by Use Case

### Production Setup (High Quality)
```bash
# Best quality retrieval and reasoning
CLIP_MODEL=sentence-transformers/clip-ViT-L-14
EMBEDDING_DIM=768
GEMINI_MODEL=gemini-1.5-flash
LLM_TEMPERATURE=0.2
```

### Development Setup (Faster)
```bash
# Faster for iteration
CLIP_MODEL=sentence-transformers/clip-ViT-B-32
EMBEDDING_DIM=512
GEMINI_MODEL=gemini-2.5-flash-lite
LLM_TEMPERATURE=0.2
```

### Cost-Optimized Setup
```bash
# Balance of cost and quality
CLIP_MODEL=sentence-transformers/clip-ViT-B-32
EMBEDDING_DIM=512
GEMINI_MODEL=gemini-1.5-flash
LLM_TEMPERATURE=0.2
```

## Embedding Dimension Migration

### Upgrading from 512 to 768 dimensions

If you have an existing database with 512-dimensional embeddings:

1. **Backup your data** (critical!)
```bash
docker compose exec db pg_dump -U $DB_USER $DB_NAME > backup.sql
```

2. **Run migration script**
```bash
docker compose exec db psql -U $DB_USER -d $DB_NAME -f /docker-entrypoint-initdb.d/migration_upgrade_to_768.sql
```

3. **Update environment variables**
```bash
CLIP_MODEL=sentence-transformers/clip-ViT-L-14
EMBEDDING_DIM=768
```

4. **Re-ingest all documents** with new model
```bash
# Example
python ingestion/ingest_unified.py path/to/document.pdf
```

### Downgrading from 768 to 512 dimensions

Follow similar steps but use `migration_downgrade_to_512.sql` (if needed).

## Verification

After configuration, verify your setup:

### 1. Check Database Connection
```bash
docker compose exec db psql -U $DB_USER -d $DB_NAME -c "SELECT version();"
```

### 2. Verify Embedding Model
```python
from ingestion.embeddings import get_clip_model, EMBEDDING_DIM
model = get_clip_model()
print(f"Model loaded: {model}")
print(f"Embedding dimensions: {EMBEDDING_DIM}")
```

### 3. Test LLM Connection
```python
from inference.llm import call_llm
response = call_llm("You are a test assistant.", [{"role": "user", "content": "Hello"}])
print(f"LLM response: {response}")
```

### 4. Verify Schema
```bash
docker compose exec db psql -U $DB_USER -d $DB_NAME -c "\d chunks"
# Should show emb column with vector(768) or vector(512)
```

## Troubleshooting

### Issue: "Embedding dimension mismatch"
**Solution**: Ensure `EMBEDDING_DIM` matches your database schema:
- If database has `vector(512)`, use `EMBEDDING_DIM=512`
- If database has `vector(768)`, use `EMBEDDING_DIM=768`
- Or run migration script to update database

### Issue: "CLIP model not found"
**Solution**: Install sentence-transformers:
```bash
pip install sentence-transformers
```

### Issue: "Gemini API key invalid"
**Solution**: 
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set in `.env`: `GEMINI_API_KEY=your_key_here`

### Issue: "Token limit exceeded"
**Solution**: 
- CLIP models have 77 token limit (inherent to architecture)
- System automatically chunks text to fit
- If errors persist, reduce chunk size in `semantic_chunks()` function

## Security Notes

⚠️ **IMPORTANT**: 
- Never commit `.env` file to git
- Keep API keys secret
- Use different keys for dev/prod
- Rotate keys regularly
- Consider using secrets management (AWS Secrets Manager, HashiCorp Vault, etc.)

## Additional Resources

- [LLM Setup Guide](LLM_SETUP.md)
- [Embedding Options Guide](EMBEDDING_OPTIONS.md)
- [Database Reset Guide](RESET_DB.md)

