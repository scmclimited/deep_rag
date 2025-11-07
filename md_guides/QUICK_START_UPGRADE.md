# Quick Start: Upgraded Deep RAG

This guide helps you quickly start using the upgraded Deep RAG system with CLIP-ViT-L/14 (768 dims) and comprehensive logging.

---

## ⚡ 30-Second Setup

```bash
# 1. Update .env
cat >> .env << 'EOF'
# Multi-Modal Embeddings (768 dims)
CLIP_MODEL=sentence-transformers/clip-ViT-L-14
EMBEDDING_DIM=768

# LLM (1M token context)
GEMINI_MODEL=gemini-1.5-flash
EOF

# 2. Start services
docker compose up -d --build

# 3. Ingest a document (any type: PDF, TXT, PNG, JPG, JPEG)
python ingestion/ingest_unified.py path/to/document.pdf "My Document"

# 4. Query with agentic reasoning (includes logging)
docker compose exec api python -m inference.cli query-graph \
  "What are the main requirements?" --thread-id session-1
```

✓ Done! Logs are in `inference/graph/logs/`

---

## 📁 Supported File Types

| Type | Extensions | Handler | Features |
|------|-----------|---------|----------|
| **PDF** | `.pdf` | `ingest.py` | Text extraction, OCR, images |
| **Text** | `.txt`, `.md` | `ingest_text.py` | Plain text chunking |
| **Images** | `.png`, `.jpg`, `.jpeg` | `ingest_image.py` | OCR + multi-modal |

**Unified Entry Point**: `ingestion/ingest_unified.py` (auto-detects file type)

---

## 🎯 Key Commands

### Ingestion
```bash
# Any file type
python ingestion/ingest_unified.py file.pdf
python ingestion/ingest_unified.py article.txt
python ingestion/ingest_unified.py diagram.png "Diagram Title"
```

### Querying (with Logging)
```bash
# LangGraph pipeline (includes agentic reasoning logs)
python inference/cli.py query-graph "Your question?" --thread-id session-1

# Direct pipeline (faster, no logs)
python inference/cli.py query "Your question?"
```

### View Logs
```bash
# Human-readable logs
cat inference/graph/logs/agent_log_*.txt

# CSV logs for training
head -20 inference/graph/logs/agent_log_*.csv
```

---

## 🔧 Configuration Options

### Production (Best Quality)
```bash
CLIP_MODEL=sentence-transformers/clip-ViT-L-14
EMBEDDING_DIM=768
GEMINI_MODEL=gemini-1.5-flash
```

### Development (Faster)
```bash
CLIP_MODEL=sentence-transformers/clip-ViT-B-32
EMBEDDING_DIM=512
GEMINI_MODEL=gemini-2.5-flash-lite
```

---

## 📊 What's Logged

Every query using `query-graph` or `infer-graph` logs:
- ✓ Question and generated plan
- ✓ Retrieval queries and results
- ✓ Retrieved chunks (IDs, pages, scores)
- ✓ Confidence evaluations
- ✓ Refinement decisions
- ✓ Final answer with citations

**Files**:
- `agent_log_TIMESTAMP.csv` - Structured data for SFT training
- `agent_log_TIMESTAMP.txt` - Human-readable for presentations

---

## 🚀 Upgrade Existing Database

```bash
# 1. Backup
docker compose exec db pg_dump -U $DB_USER $DB_NAME > backup.sql

# 2. Migrate
docker compose exec db psql -U $DB_USER -d $DB_NAME \
  -f /docker-entrypoint-initdb.d/migration_upgrade_to_768.sql

# 3. Update .env
CLIP_MODEL=sentence-transformers/clip-ViT-L-14
EMBEDDING_DIM=768

# 4. Re-ingest all documents
python ingestion/ingest_unified.py document.pdf
```

---

## 🆘 Troubleshooting

### "Embedding dimension mismatch"
→ Ensure `.env` matches database schema:
- Database `vector(768)` → `EMBEDDING_DIM=768`
- Database `vector(512)` → `EMBEDDING_DIM=512`

### "File type not supported"
→ Check supported extensions:
```python
from ingestion.ingest_unified import list_supported_extensions
print(list_supported_extensions())
```

### "Token limit exceeded"
→ CLIP has 77 token limit. System auto-truncates. If errors persist:
- Check logs for specific failure
- Reduce chunk size in `semantic_chunks()`

---

## 📚 Full Documentation

- [Complete README](README.md)
- [Upgrade Summary](UPGRADE_SUMMARY.md)
- [Environment Setup](md_guides/ENVIRONMENT_SETUP.md)
- [LLM Setup](md_guides/LLM_SETUP.md)

---

## ✨ Key Improvements

| Feature | Before | After |
|---------|--------|-------|
| **Embedding Dimensions** | 512 | **768** (+50%) |
| **Model** | CLIP-ViT-B/32 | **CLIP-ViT-L/14** |
| **Retrieval Quality** | Good | **Better** |
| **Reasoning Logs** | None | **CSV + TXT** |
| **File Type Support** | Manual routing | **Auto-detect** |
| **Documentation** | Basic | **Comprehensive** |

---

**Ready to go!** 🚀

For detailed information, see [UPGRADE_SUMMARY.md](UPGRADE_SUMMARY.md)

