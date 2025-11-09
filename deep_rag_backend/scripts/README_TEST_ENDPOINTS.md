# Endpoint Testing Scripts

This directory contains comprehensive test scripts for all Deep RAG endpoints.

## Scripts

### 1. `test_endpoints_make.sh`
Tests all endpoints using Make commands (CLI interface).

**Usage:**
```bash
# Make sure services are running (from project root)
make up

# Run the test script (from deep_rag_backend directory)
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

**Usage:**
```bash
# Make sure services are running (from project root)
make up

# Run the test script (from deep_rag_backend directory)
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

## Prerequisites

1. **Services must be running:**
   ```bash
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

| Endpoint | Method | Pipeline | Flags Tested |
|----------|--------|----------|--------------|
| `/ingest` | POST | N/A | `title` |
| `/ask` | POST | Direct | `doc_id`, `cross_doc` |
| `/ask-graph` | POST | LangGraph | `doc_id`, `cross_doc`, `thread_id` |
| `/infer` | POST | Direct | `attachment`, `title`, `cross_doc` |
| `/infer-graph` | POST | LangGraph | `attachment`, `title`, `thread_id`, `cross_doc` |
| `/health` | GET | N/A | None |

### File Type Coverage

- ✅ PDF files (`.pdf`)
- ✅ Image files (`.png`, `.jpg`, `.jpeg`)

### Configuration Coverage

- ✅ Query all documents (no `doc_id`)
- ✅ Query specific document (with `doc_id`)
- ✅ Cross-document retrieval (`cross_doc=true`)
- ✅ Thread tracking (`thread_id`)
- ✅ Ingest + Query (with `attachment`)
- ✅ Query only (no `attachment`)

## Logging Verification

After running the tests, verify logging:

### 1. Check API Logs
```bash
# View container logs
docker compose logs api

# Or tail logs
make logs
```

### 2. Check Graph Logs (LangGraph endpoints)
```bash
# List production log files (from project root)
ls -la deep_rag_backend/inference/graph/logs/

# List test log files (from project root)
ls -la deep_rag_backend/inference/graph/logs/test_logs/

# View latest production log
tail -f deep_rag_backend/inference/graph/logs/agent_log_*.txt

# View latest test log
tail -f deep_rag_backend/inference/graph/logs/test_logs/agent_log_*.txt
```

### 3. Check Thread Tracking Table
```bash
# Query thread_tracking table
docker compose exec db psql -U $DB_USER -d $DB_NAME -c "SELECT thread_id, query_text, doc_ids, entry_point, pipeline_type, cross_doc, created_at FROM thread_tracking ORDER BY created_at DESC LIMIT 10;"
```

## Expected Output

### Make Script Output
- Each test shows the command being executed
- Results are displayed inline
- `doc_id` is extracted from ingest responses for subsequent queries
- Summary at the end confirms all tests completed

### REST API Script Output
- Each test shows the curl command
- JSON responses are formatted (if `jq` is available)
- `doc_id` is extracted from ingest responses for subsequent queries
- Summary at the end confirms all tests completed

## Troubleshooting

### Script fails with "command not found"
- Make sure scripts are executable: `chmod +x deep_rag_backend/scripts/test_endpoints_*.sh`
- Run scripts from `deep_rag_backend` directory (or use full path from project root)

### "Connection refused" errors
- Ensure services are running: `make up`
- Check API is healthy: `curl http://localhost:8000/health`

### "File not found" errors
- Verify sample files exist in `deep_rag_backend/inference/samples/`
- Check file paths in scripts match your file structure
- Run scripts from `deep_rag_backend` directory (scripts use relative paths from that directory)

### No `doc_id` extracted
- Check ingest endpoint responses
- Verify JSON parsing (if using `jq`)
- Scripts will skip `doc_id`-specific tests if extraction fails

## Next Steps

After running tests:

1. **Verify all endpoints responded correctly**
2. **Check logs for any errors or warnings**
3. **Verify thread_tracking table has entries**
4. **Check graph logs for LangGraph endpoints**
5. **Review response formats and data**

## Notes

- Scripts use sample files from `deep_rag_backend/inference/samples/`
- Run scripts from `deep_rag_backend` directory
- `doc_id` is extracted from first ingest response for subsequent queries
- `thread_id` is generated with timestamp for uniqueness
- All tests are run sequentially (not parallelized)
- Scripts exit on first error (`set -e`)
- Test logs are written to `deep_rag_backend/inference/graph/logs/test_logs/` (ignored by git)

