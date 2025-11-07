# Endpoint Testing Guide

This guide explains how to test all Deep RAG endpoints to verify they work correctly and log properly.

## Quick Start

### 1. Start Services
```bash
make up
```

### 2. Run Tests

**Option A: Quick Test (Recommended for first run)**
```bash
make test-endpoints-quick
```
Tests one example of each endpoint type - fastest way to verify everything works.

**Option B: Full Test Suite**
```bash
make test-endpoints
```
Tests all endpoints via both Make commands and REST API - comprehensive coverage.

**Option C: Individual Test Scripts**
```bash
# Test via Make commands only
make test-endpoints-make

# Test via REST API only
make test-endpoints-rest

# Or run scripts directly
./scripts/test_endpoints_make.sh
./scripts/test_endpoints_rest.sh
./scripts/test_endpoints_quick.sh
```

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

## Test Scripts

### 1. `test_endpoints_make.sh`
Tests all endpoints using Make commands (CLI interface).

**Features:**
- Tests ingest, query, query-graph, infer, and infer-graph endpoints
- Extracts `doc_id` from ingest responses for subsequent queries
- Tests all flag combinations (doc_id, cross_doc, thread_id)
- Uses sample files from `inference/samples/`

**Usage:**
```bash
make test-endpoints-make
# Or directly:
./scripts/test_endpoints_make.sh
```

### 2. `test_endpoints_rest.sh`
Tests all endpoints using REST API (curl requests).

**Features:**
- Tests all REST endpoints with curl
- Formats JSON responses (if `jq` is available)
- Tests all flag combinations
- Includes health check endpoint

**Usage:**
```bash
make test-endpoints-rest
# Or directly:
./scripts/test_endpoints_rest.sh
```

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
make test-endpoints-quick
# Or directly:
./scripts/test_endpoints_quick.sh
```

## Verifying Logging

After running tests, verify that logging is working correctly:

### 1. Check API Logs
```bash
# View container logs
docker compose logs api

# Or tail logs
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
# List log files
ls -la inference/graph/logs/

# View latest log
tail -f inference/graph/logs/agent_log_*.txt

# Or view in Docker
docker compose exec api ls -la /app/inference/graph/logs/
```

**What to look for:**
- ✅ Agent step logs (planner, retriever, compressor, critic, synthesizer)
- ✅ Confidence scores
- ✅ Evidence retrieval counts
- ✅ Query refinements (if confidence < threshold)
- ✅ Final answers with citations

### 3. Check Thread Tracking Table
```bash
# Query thread_tracking table
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
# Verify all tables exist
docker compose exec db psql -U $DB_USER -d $DB_NAME -c "
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('documents', 'chunks', 'thread_tracking')
ORDER BY table_name;
"
```

## Expected Responses

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

## Troubleshooting

### Script fails with "command not found"
- Make sure scripts are executable: `chmod +x scripts/test_endpoints_*.sh`
- Run from project root directory
- On Windows, use Git Bash or WSL

### "Connection refused" errors
- Ensure services are running: `make up`
- Check API is healthy: `curl http://localhost:8000/health`
- Wait a few seconds after `make up` for services to start

### "File not found" errors
- Verify sample files exist in `inference/samples/`
- Check file paths in scripts match your file structure
- On Windows, ensure paths use forward slashes for Docker

### No `doc_id` extracted
- Check ingest endpoint responses
- Verify JSON parsing (if using `jq`)
- Scripts will skip `doc_id`-specific tests if extraction fails
- Check API logs for ingest errors

### No thread_tracking entries
- Verify `thread_tracking` table exists: `make test DOCKER=true`
- Check API logs for errors
- LangGraph endpoints should always create entries
- Direct pipeline endpoints may not create entries (depends on implementation)

### Logs not appearing
- Check log directory exists: `ls -la inference/graph/logs/`
- Verify volume mount in `docker-compose.yml`
- Check file permissions
- LangGraph endpoints create logs in `inference/graph/logs/`

## Sample Files

The test scripts use these sample files (must exist):
- `inference/samples/NYMBL - AI Engineer - Omar.pdf` - PDF document
- `inference/samples/technical_assessment_brief_1.png` - Image file
- `inference/samples/technical_assessment_brief_2.png` - Image file

If these files don't exist, update the file paths in the test scripts or add your own sample files.

## Next Steps

After running tests:

1. ✅ **Verify all endpoints responded correctly**
2. ✅ **Check logs for any errors or warnings**
3. ✅ **Verify thread_tracking table has entries**
4. ✅ **Check graph logs for LangGraph endpoints**
5. ✅ **Review response formats and data**
6. ✅ **Verify citations include full doc_id**
7. ✅ **Check cross-doc flag behavior**

## Notes

- Scripts use sample files from `inference/samples/`
- `doc_id` is extracted from first ingest response for subsequent queries
- `thread_id` is generated with timestamp for uniqueness
- All tests are run sequentially (not parallelized)
- Scripts exit on first error (`set -e`)
- On Windows, use Git Bash or WSL to run bash scripts

