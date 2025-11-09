# Endpoint Testing Guide

This guide explains how to test all Deep RAG endpoints to verify they work correctly and log properly.

## Quick Start

### 1. Start Services
```bash
# From project root (deep_rag/)
make up
```

### 2. Run Tests

**Option A: Quick Test (Recommended for first run)**
```bash
# From project root (deep_rag/)
make test-endpoints-quick
```
Tests one example of each endpoint type - fastest way to verify everything works.

**Option B: Full Test Suite**
```bash
# From project root (deep_rag/)
make test-endpoints
```
Tests all endpoints via both Make commands and REST API - comprehensive coverage.

**Option C: Automatic on Boot**
```bash
# Set in .env (project root)
AUTOMATE_ENDPOINT_RUNS_ON_BOOT=true

# Then run (from project root)
make up-and-test
```
This will automatically run `make test-endpoints` (full suite: Make + REST API) after unit and integration tests complete.

**Option D: Individual Test Scripts**
```bash
# From project root (deep_rag/)
make test-endpoints-make   # Test via Make commands only
make test-endpoints-rest   # Test via REST API only

# Or run scripts directly (from deep_rag_backend directory)
cd deep_rag_backend
./scripts/test_endpoints_make.sh
./scripts/test_endpoints_rest.sh
./scripts/test_endpoints_quick.sh
```

## Prerequisites

1. **Services must be running:**
   ```bash
   # From project root (deep_rag/)
   make up
   ```

2. **Sample files must exist:**
   - `deep_rag_backend/inference/samples/NYMBL - AI Engineer - Omar.pdf`
   - `deep_rag_backend/inference/samples/technical_assessment_brief_1.png`
   - `deep_rag_backend/inference/samples/technical_assessment_brief_2.png`

3. **Optional tools (for better output):**
   - `jq` - For JSON formatting (install: `apt-get install jq` or `brew install jq`)
   - If `jq` is not available, scripts will still work but output won't be formatted

## What Gets Tested

### Endpoint Coverage

| Endpoint | Method | Pipeline | Tested Configurations |
|----------|--------|----------|----------------------|
| `/ingest` | POST | N/A | PDF, Image |
| `/ask` | POST | Direct | All docs, Specific doc_id, Cross-doc |
| `/ask-graph` | POST | LangGraph | All docs, Specific doc_id, Cross-doc, Thread ID |
| `/infer` | POST | Direct | PDF+Query, Image+Query, Cross-doc, Query-only |
| `/infer-graph` | POST | LangGraph | PDF+Query, Image+Query, Cross-doc, Thread ID, Query-only |
| `/health` | GET | N/A | Basic health check |

### File Type Coverage

- ✅ **PDF files** (`.pdf`) - Full text extraction with OCR fallback
- ✅ **Image files** (`.png`, `.jpg`, `.jpeg`) - Image captioning via OCR/vision

### Configuration Coverage

- ✅ **Query all documents** (no `doc_id` filter)
- ✅ **Query specific document** (with `doc_id` filter)
- ✅ **Cross-document retrieval** (`cross_doc=true`)
- ✅ **Thread tracking** (`thread_id` for LangGraph endpoints)
- ✅ **Ingest + Query** (with `attachment` file)
- ✅ **Query only** (no `attachment` file)

<details>
<summary><strong>Test Scripts</strong> - Click to expand</summary>

This directory (`deep_rag_backend/scripts/`) contains comprehensive test scripts for all Deep RAG endpoints.

### 1. `test_endpoints_make.sh`
Tests all endpoints using Make commands (CLI interface).

**Features:**
- Tests ingest, query, query-graph, infer, and infer-graph endpoints
- Extracts `doc_id` from ingest responses for subsequent queries
- Tests all flag combinations (doc_id, cross_doc, thread_id)
- Uses sample files from `deep_rag_backend/inference/samples/`

**Usage:**
```bash
# From project root (deep_rag/)
make test-endpoints-make

# Or directly (from deep_rag_backend directory)
cd deep_rag_backend
./scripts/test_endpoints_make.sh
```

**What it tests:**
- ✅ Ingest endpoints (PDF, Image)
- ✅ Query endpoints (Direct pipeline - all docs, specific doc_id, cross-doc)
- ✅ Query-graph endpoints (LangGraph pipeline - all docs, specific doc_id, cross-doc, thread_id)
- ✅ Infer endpoints (Direct pipeline - ingest + query with PDF, Image, cross-doc)
- ✅ Infer-graph endpoints (LangGraph pipeline - ingest + query with PDF, Image, cross-doc, thread_id)

### 2. `test_endpoints_rest.sh`
Tests all endpoints using REST API (curl requests).

**Features:**
- Tests all REST endpoints with curl
- Formats JSON responses (if `jq` is available)
- Tests all flag combinations
- Includes health check endpoint

**Usage:**
```bash
# From project root (deep_rag/)
make test-endpoints-rest

# Or directly (from deep_rag_backend directory)
cd deep_rag_backend
./scripts/test_endpoints_rest.sh
```

**What it tests:**
- ✅ Health check endpoint
- ✅ Ingest endpoints (PDF, Image)
- ✅ Ask endpoints (Direct pipeline - all docs, specific doc_id, cross-doc)
- ✅ Ask-graph endpoints (LangGraph pipeline - all docs, specific doc_id, cross-doc, thread_id)
- ✅ Infer endpoints (Direct pipeline - ingest + query with PDF, Image, cross-doc, query-only)
- ✅ Infer-graph endpoints (LangGraph pipeline - ingest + query with PDF, Image, cross-doc, thread_id, query-only)

**Prerequisites:**
- Optional: `jq` for JSON formatting (`apt-get install jq` or `brew install jq`)
- Scripts work without `jq`, but output won't be formatted

### 3. `test_endpoints_quick.sh`
Quick test - one example of each endpoint type.

**Features:**
- Fastest way to verify all endpoints work
- Tests one example of each type
- Good for quick verification after changes

**Usage:**
```bash
# From project root (deep_rag/)
make test-endpoints-quick

# Or directly (from deep_rag_backend directory)
cd deep_rag_backend
./scripts/test_endpoints_quick.sh
```

</details>

<details>
<summary><strong>Verifying Logging</strong> - Click to expand</summary>

After running tests, verify that logging is working correctly:

### 1. Check API Logs
```bash
# View container logs (from project root)
docker compose logs api

# Or tail logs (from project root)
make logs
```

**What to look for:**
- ✅ Ingest success messages: `✅ Ingested PDF: ...`
- ✅ Document ID logging: `📋 Document ID: ...`
- ✅ Query logging: `Querying with document filter: ...`
- ✅ Cross-doc logging: `Cross-document retrieval enabled`
- ✅ Error messages (if any)

### 2. Check Graph Logs (LangGraph Endpoints)
```bash
# List production log files (from project root)
ls -la deep_rag_backend/inference/graph/logs/

# List test log files (from project root)
ls -la deep_rag_backend/inference/graph/logs/test_logs/

# View latest production log (from project root)
tail -f deep_rag_backend/inference/graph/logs/agent_log_*.txt

# View latest test log (from project root)
tail -f deep_rag_backend/inference/graph/logs/test_logs/agent_log_*.txt

# Or view in Docker (from project root)
docker compose exec api ls -la /app/inference/graph/logs/
docker compose exec api ls -la /app/inference/graph/logs/test_logs/
```

**What to look for:**
- ✅ Agent step logs (planner, retriever, compressor, critic, synthesizer)
- ✅ Confidence scores
- ✅ Evidence retrieval counts
- ✅ Query refinements (if confidence < threshold)
- ✅ Final answers with citations

### 3. Check Thread Tracking Table
```bash
# Query thread_tracking table (from project root)
docker compose exec db psql -U $DB_USER -d $DB_NAME -c "
SELECT 
    thread_id,
    entry_point,
    pipeline_type,
    cross_doc,
    array_length(doc_ids, 1) as doc_count,
    length(query_text) as query_length,
    length(final_answer) as answer_length,
    created_at
FROM thread_tracking 
ORDER BY created_at DESC 
LIMIT 10;
"
```

**What to look for:**
- ✅ Entries for each query (especially LangGraph endpoints)
- ✅ Correct `entry_point` (`rest` for REST API, `make` for Make commands)
- ✅ Correct `pipeline_type` (`direct` or `langgraph`)
- ✅ `cross_doc` flag matches request
- ✅ `doc_ids` array populated (if documents were retrieved)
- ✅ `query_text` and `final_answer` populated

### 4. Check Database Schema
```bash
# Verify all tables exist (from project root)
docker compose exec db psql -U $DB_USER -d $DB_NAME -c "
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('documents', 'chunks', 'thread_tracking')
ORDER BY table_name;
"
```

</details>

<details>
<summary><strong>Expected Responses</strong> - Click to expand</summary>

### Ingest Endpoint (`/ingest`)
```json
{
  "status": "success",
  "filename": "document.pdf",
  "file_type": ".pdf",
  "title": "Document Title",
  "doc_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Ask Endpoint (`/ask`)
```json
{
  "answer": "The answer with citations...",
  "mode": "query_only",
  "pipeline": "direct",
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "cross_doc": false
}
```

**Note:** `doc_id` may be `null` if not provided in request.

### Ask-Graph Endpoint (`/ask-graph`)
```json
{
  "answer": "The answer with citations...",
  "mode": "query_only",
  "pipeline": "langgraph",
  "thread_id": "test-thread-1234567890",
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "cross_doc": false
}
```

**Note:** `doc_id` may be `null` if not provided in request. `thread_id` defaults to `"default"` if not provided.

### Infer Endpoint (`/infer`)
```json
{
  "answer": "The answer with citations...",
  "mode": "ingest_and_query",
  "pipeline": "direct",
  "attachment_processed": true,
  "filename": "document.pdf",
  "file_type": ".pdf",
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "cross_doc": false
}
```

**Note:** When `attachment` is not provided, response will have:
- `"mode": "query_only"`
- `"attachment_processed": false`
- `"filename": null`
- `"file_type": null`
- `"doc_id": null`

### Infer-Graph Endpoint (`/infer-graph`)
```json
{
  "answer": "The answer with citations...",
  "mode": "ingest_and_query",
  "pipeline": "langgraph",
  "thread_id": "test-thread-1234567890",
  "attachment_processed": true,
  "filename": "document.pdf",
  "file_type": ".pdf",
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "cross_doc": false
}
```

**Note:** When `attachment` is not provided, response will have:
- `"mode": "query_only"`
- `"attachment_processed": false`
- `"filename": null`
- `"file_type": null`
- `"doc_id": null`
- `thread_id` defaults to `"default"` if not provided

</details>

<details>
<summary><strong>Troubleshooting</strong> - Click to expand</summary>

### Script fails with "command not found"
- Make sure scripts are executable: `chmod +x deep_rag_backend/scripts/test_endpoints_*.sh`
- Run scripts from `deep_rag_backend` directory (or use full path from project root)
- On Windows, use Git Bash or WSL

### "Connection refused" errors
- Ensure services are running: `make up` (from project root)
- Check API is healthy: `curl http://localhost:8000/health`
- Wait a few seconds after `make up` for services to start

### "File not found" errors
- Verify sample files exist in `deep_rag_backend/inference/samples/`
- Check file paths in scripts match your file structure
- Run scripts from `deep_rag_backend` directory (scripts use relative paths from that directory)
- On Windows, ensure paths use forward slashes for Docker

### No `doc_id` extracted
- Check ingest endpoint responses
- Verify JSON parsing (if using `jq`)
- Scripts will skip `doc_id`-specific tests if extraction fails
- Check API logs for ingest errors

### No thread_tracking entries
- Verify `thread_tracking` table exists: `make test DOCKER=true` (from project root)
- Check API logs for errors
- LangGraph endpoints should always create entries
- Direct pipeline endpoints may not create entries (depends on implementation)

### Logs not appearing
- Check production log directory exists: `ls -la deep_rag_backend/inference/graph/logs/` (from project root)
- Check test log directory exists: `ls -la deep_rag_backend/inference/graph/logs/test_logs/` (from project root)
- Verify volume mount in `docker-compose.yml` (project root)
- Check file permissions
- LangGraph endpoints create production logs in `deep_rag_backend/inference/graph/logs/`
- Test executions create logs in `deep_rag_backend/inference/graph/logs/test_logs/`

</details>

## Sample Files

The test scripts use these sample files (must exist in `deep_rag_backend/inference/samples/`):
- `deep_rag_backend/inference/samples/NYMBL - AI Engineer - Omar.pdf` - PDF document
- `deep_rag_backend/inference/samples/technical_assessment_brief_1.png` - Image file
- `deep_rag_backend/inference/samples/technical_assessment_brief_2.png` - Image file

If these files don't exist, update the file paths in the test scripts or add your own sample files.

<details>
<summary><strong>Next Steps</strong> - Click to expand</summary>

After running tests:

1. ✅ **Verify all endpoints responded correctly**
2. ✅ **Check logs for any errors or warnings**
3. ✅ **Verify thread_tracking table has entries**
4. ✅ **Check graph logs for LangGraph endpoints**
5. ✅ **Review response formats and data**
6. ✅ **Verify citations include full doc_id**
7. ✅ **Check cross-doc flag behavior**

</details>

<details>
<summary><strong>Notes</strong> - Click to expand</summary>

- Scripts are located in `deep_rag_backend/scripts/`
- Scripts use sample files from `deep_rag_backend/inference/samples/`
- Run scripts from `deep_rag_backend` directory (or use Make commands from project root)
- `doc_id` is extracted from first ingest response for subsequent queries
- `thread_id` is generated with timestamp for uniqueness
- All tests are run sequentially (not parallelized)
- Scripts exit on first error (`set -e`)
- Test logs are written to `deep_rag_backend/inference/graph/logs/test_logs/` (ignored by git)
- On Windows, use Git Bash or WSL to run bash scripts

</details>
