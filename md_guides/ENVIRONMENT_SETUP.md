# Environment Configuration Guide

This guide explains all environment variables used in Deep RAG.

## Core Configuration (.env)

Create a `.env` file in the project root with the following variables:

```bash
# =============================================================================
# Deep RAG Backend - Environment Configuration
# =============================================================================
# This file contains environment variables specific to the Backend API service.
# The backend provides FastAPI endpoints for document ingestion and querying.
#
# Copy this file to .env and fill in your values:
#   cp .env.example .env
# =============================================================================

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
# Database connection settings for PostgreSQL with pgvector extension.
# The backend connects to the database to store and retrieve document embeddings.

DB_HOST=localhost
DB_PORT=5432
DB_USER=rag
DB_PASS=rag
DB_NAME=deep_rag_db

# =============================================================================
# OPTIONAL CONFIGURATION
# =============================================================================

# Run tests on backend startup
# Set to true to run database schema tests when backend container starts
RUN_TESTS_ON_STARTUP=false
AUTOMATE_ENDPOINT_RUNS_ON_BOOT=false

# =============================================================================
# EMBEDDING CONFIGURATION - Multi-modal embedding model settings for document and image embeddings.
# =============================================================================

# CLIP Model Selection
#   - openai/clip-vit-large-patch14-336 (RECOMMENDED) - 768 dimensions, better quality
#   - sentence-transformers/clip-ViT-B-32 - 512 dimensions, faster performance
# CLIP_MODEL=openai/clip-vit-large-patch14-336
CLIP_MODEL=openai/clip-vit-large-patch14-336
CLIP_MODEL_PATH=/app/models/{CLIP_MODEL}

# Reranker model (chunk reranker)
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_MODEL_PATH=/app/models/{RERANK_MODEL}

# Embedding Dimensions (must match CLIP model)
#   - 768 for CLIP-ViT-L-14
#   - 512 for CLIP-ViT-B-32
EMBEDDING_DIM=768

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
# Language Model provider settings for agentic reasoning and query processing.

# LLM Provider (currently supports: gemini, openai, ollama)
LLM_PROVIDER=gemini

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Gemini Model Selection
# Recommended models:
#   - gemini-1.5-flash (RECOMMENDED) - 1M token context, fast, excellent reasoning
#   - gemini-2.0-flash (LATEST) - 1M token context, improved performance
#   - gemini-2.5-flash-lite (LIGHTWEIGHT) - Limited context, very fast
GEMINI_MODEL=gemini-2.0-flash


# LLM Temperature (0.0 = deterministic, 1.0 = creative)
# Lower values (0.1-0.3) are recommended for RAG applications
LLM_TEMPERATURE=0.15

# =============================================================================
# DO NOT TOUCH - FINE-TUNED RETRIEVAL/CONFIDENCE THRESHOLDS PARAMETERS
# =============================================================================

# Retrieval thresholds
K_RETRIEVER=8
K_CRITIC=6 
K_LEX=60
K_VEC=60

# Confidence scoring thresholds
CONF_W0=-0.5          # Bias (less negative = higher base)
CONF_W1=2.4           # max_rerank (reduced from 3.0)
CONF_W2=1.1           # margin (reduced from 1.5)
CONF_W3=1.6           # mean_cosine (reduced from 2.2)
CONF_W4=-0.4          # cosine_std (more negative)
CONF_W5=0.8           # cos_coverage (reduced from 1.0)
CONF_W6=1.3           # bm25_norm (reduced from 1.5)
CONF_W7=1.1           # term_coverage (reduced from 1.2)
CONF_W8=0.6           # unique_page_frac (reduced from 0.8)
CONF_W9=0.45          # doc_diversity (increased from 0.4)
CONF_W10=1.25         # answer_overlap (reduced from 1.2)

# Decision thresholds
CONF_ABSTAIN_TH=0.20  # 35%
CONF_CLARIFY_TH=0.60  # 52%
MAX_ITERS=3           # Increased for better convergence on complex multi-document queries
THRESH=.30            # matches 50% CE/lex+vec heuristic (used by critic and synthesizer)

# Synthesizer Confidence Thresholds (percentage, 0-100)
# These control when the synthesizer abstains BEFORE calling the LLM
# NOTE: These are OPTIONAL - if not set, defaults are used automatically
SYNTHESIZER_CONFIDENCE_THRESHOLD_DEFAULT=40.0          # Default threshold for general queries
                                                        # If not set, defaults to 40.0%
SYNTHESIZER_CONFIDENCE_THRESHOLD_EXPLICIT_SELECTION=30.0  # Lower threshold when docs explicitly selected/attached
                                                          # If not set, defaults to THRESH * 100 (30.0%)
                                                          # DO NOT use {THRESH} placeholder - use numeric value (e.g., 30.0)
```

## Synthesizer Confidence Thresholds

The synthesizer uses confidence thresholds to determine whether to call the LLM or abstain early:

- **`SYNTHESIZER_CONFIDENCE_THRESHOLD_DEFAULT`** (default: `40.0%`): Used for general queries and cross-document search without explicit document selection
- **`SYNTHESIZER_CONFIDENCE_THRESHOLD_EXPLICIT_SELECTION`** (default: `THRESH * 100 = 30.0%`): Used when documents are explicitly selected (`selected_doc_ids`), attached (`uploaded_doc_ids`), or provided via `doc_id`

**How it works:**
- If confidence is below the threshold, the synthesizer abstains BEFORE calling the LLM (saves tokens)
- If confidence is above the threshold, the LLM is called
- The LLM can still return "I don't know" even if confidence is above threshold (detected by `citation_pruner`)

**Threshold Selection Logic:**
- Explicit selection threshold (30%) is used when:
  - `selected_doc_ids` is provided (user selected documents in UI)
  - `uploaded_doc_ids` is provided (user attached/uploaded documents)
  - `doc_id` is provided (document from ingestion/previous query)
  - AND `cross_doc=False` OR `cross_doc=True` with specific docs selected (hybrid mode)
- Default threshold (40%) is used when:
  - `cross_doc=True` AND no specific documents are selected
  - No documents are explicitly provided

**Note:** `SYNTHESIZER_CONFIDENCE_THRESHOLD_EXPLICIT_SELECTION` defaults to `THRESH * 100` (30.0%) if not explicitly set, ensuring consistency with the critic's chunk strength threshold.

## Configuration by Use Case

### Production Setup (High Quality)
```bash
# Best quality retrieval and reasoning
CLIP_MODEL=openai/clip-vit-large-patch14-336
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
CLIP_MODEL=openai/clip-vit-large-patch14-336
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
**Solution**: Install transformers:
```bash
pip install transformers
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

## Testing Configuration

### Startup Tests

**`RUN_TESTS_ON_STARTUP`** (default: `false`)
- Runs database schema tests when the API container starts
- Verifies all required tables exist (`documents`, `chunks`, `thread_tracking`)
- Ensures database schema is properly initialized
- Adds a few seconds to container startup time

**To enable:**
```bash
RUN_TESTS_ON_STARTUP=true
```

### Endpoint Tests on Boot

**`AUTOMATE_ENDPOINT_RUNS_ON_BOOT`** (default: `false`)
- Runs endpoint tests after `make up-and-test` completes
- Executes `make test-endpoints` (full suite: Make + REST API) to verify all endpoints work
- Tests all endpoint configurations (ingest, query, infer) via both Make commands and REST API
- Adds a few minutes to the startup process but provides comprehensive verification

**To enable:**
```bash
AUTOMATE_ENDPOINT_RUNS_ON_BOOT=true
```

**What gets tested:**
- ✅ Health check endpoint
- ✅ Ingest endpoints (PDF, Image)
- ✅ Query endpoints (Direct pipeline)
- ✅ Query-graph endpoints (LangGraph pipeline)
- ✅ Infer endpoints (Direct pipeline)
- ✅ Infer-graph endpoints (LangGraph pipeline)

**Note:** Both testing options are **optional** and disabled by default. Enable them for comprehensive verification during development or CI/CD pipelines.

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

