# Performance Fixes for Concurrent Query Handling

## Problem
- **Thread B**: Timeout after 120 seconds during cross-doc query
- **Thread A**: Network error during query with attachments
- **Root Cause**: Resource contention, connection exhaustion, and timeout limits

## Docker Stats Analysis
Your containers were using **very little** of available resources:
- DB: 51MB / 2GB (2.5% memory)
- API: 1.9GB / 4GB (47% memory)
- Frontend: 24MB / 62GB (negligible)

**Available Hardware**: 128GB RAM, 32 CPUs, 16GB VRAM

## Fixes Applied

### 1. Increased Docker Resource Limits ✅

**Database Container:**
- CPUs: 4 → **16** (4x increase)
- Memory: 2GB → **16GB** (8x increase)
- Max connections: 100 → **200**

**API Container:**
- CPUs: 4 → **16** (4x increase)
- Memory: 4GB → **32GB** (8x increase)
- Uvicorn workers: 4 → **8** (2x increase)
- Added `UVICORN_TIMEOUT_KEEP_ALIVE: 300` (5 minutes)

### 2. PostgreSQL Performance Tuning ✅

Created `vector_db/postgresql.conf` with optimized settings:

```ini
# Connection Settings
max_connections = 200

# Memory Settings
shared_buffers = 4GB                    # 25% of available RAM
effective_cache_size = 12GB             # 75% of available RAM
work_mem = 64MB                         # Per operation
maintenance_work_mem = 1GB              # For VACUUM, CREATE INDEX

# Parallel Query Settings
max_worker_processes = 16
max_parallel_workers_per_gather = 4
max_parallel_workers = 16

# SSD Optimization
random_page_cost = 1.1                  # Default 4.0 for HDD
effective_io_concurrency = 200

# Logging
log_min_duration_statement = 1000       # Log queries > 1 second
log_lock_waits = on                     # Debug lock contention
```

### 3. Connection Pooling ✅

**Before:**
- Every query created a new database connection
- No connection reuse
- Connection exhaustion under load

**After:**
- Implemented `psycopg2.pool.ThreadedConnectionPool`
- Pool size: 10-50 connections
- Thread-safe connection management
- Automatic connection return to pool

**File**: `deep_rag_backend/retrieval/db_utils.py`

```python
@contextmanager
def connect():
    """Returns a connection from the pool."""
    conn_pool = _get_pool()
    conn = conn_pool.getconn()
    try:
        yield conn
    finally:
        conn_pool.putconn(conn)
```

### 4. Extended Frontend Timeouts ✅

**Before:**
- Default API timeout: 2 minutes
- Long-running operations: 5 minutes

**After:**
- Default API timeout: **3 minutes**
- Long-running operations: **10 minutes**
- Cross-doc queries automatically use long-running instance

**File**: `deep_rag_frontend_vue/src/services/api.js`

```javascript
// Use long-running instance for cross-doc queries
const apiInstance = crossDoc ? apiLongRunning : api
const response = await apiInstance.post('/ask-graph', data)
```

## Expected Improvements

### Throughput
- **8x more Uvicorn workers** → 8 concurrent requests (was 4)
- **50 pooled connections** → No connection exhaustion
- **16 CPUs per container** → Better parallelism

### Latency
- **Connection pooling** → Eliminates connection overhead (~50-100ms per query)
- **PostgreSQL tuning** → Faster query execution with parallel workers
- **Increased shared_buffers** → More data cached in memory

### Reliability
- **10-minute timeouts** → Complex cross-doc queries won't timeout
- **Lock contention reduced** → Connection pool prevents database locks
- **Better resource utilization** → Containers can use full hardware capacity

## Testing Instructions

### 1. Rebuild and Restart

```bash
# Stop all containers
docker compose down

# Rebuild with new configuration
docker compose build --no-cache

# Start with new resource limits
docker compose up -d

# Watch logs
docker compose logs -f api
```

### 2. Verify PostgreSQL Configuration

```bash
# Connect to database
docker exec -it deep_rag_pgvector psql -U your_user -d your_db

# Check settings
SHOW shared_buffers;
SHOW max_connections;
SHOW max_worker_processes;
SHOW work_mem;
```

Expected output:
```
shared_buffers        | 4GB
max_connections       | 200
max_worker_processes  | 16
work_mem              | 64MB
```

### 3. Test Concurrent Queries

**Thread A**: Attach 3 documents + query  
**Thread B**: Cross-doc search (no attachments)

**Expected behavior:**
- Both queries complete successfully
- No timeout errors
- No network errors
- Logs show connection pool usage

### 4. Monitor Resource Usage

```bash
# Watch Docker stats
docker stats deep_rag_api deep_rag_pgvector

# Check PostgreSQL connections
docker exec -it deep_rag_pgvector psql -U your_user -d your_db -c "SELECT count(*) FROM pg_stat_activity;"
```

**Expected:**
- API memory usage: 4-8GB (was 1.9GB)
- DB memory usage: 4-8GB (was 51MB)
- Active connections: 10-30 (from pool)

## Troubleshooting

### If Timeouts Still Occur

1. **Check LangGraph iterations**: Look for excessive refinement loops
   ```bash
   docker compose logs api | grep "iterations="
   ```

2. **Increase timeout further**: Edit `api.js` and increase `timeout: 600000` to `900000` (15 minutes)

3. **Check database locks**:
   ```sql
   SELECT * FROM pg_locks WHERE NOT granted;
   ```

### If Memory Issues Occur

1. **Reduce pool size**: Edit `db_utils.py`, change `maxconn=50` to `maxconn=30`

2. **Reduce shared_buffers**: Edit `postgresql.conf`, change `shared_buffers = 4GB` to `2GB`

3. **Reduce Uvicorn workers**: Edit `docker-compose.yml`, change `UVICORN_WORKERS: 8` to `4`

## Performance Metrics to Track

1. **Query latency**: Time from request to response
2. **Connection pool utilization**: Active connections / max connections
3. **Database CPU usage**: Should be 10-30% for typical queries
4. **Memory usage**: API should use 4-8GB, DB should use 4-8GB
5. **Timeout rate**: Should be 0% after fixes

## Next Steps

If performance issues persist:

1. **Add query caching**: Cache frequent queries in Redis
2. **Optimize embeddings**: Use smaller model (CLIP-ViT-B/32 instead of L/14)
3. **Reduce retrieval size**: Lower `k_lex` and `k_vec` from 40 to 20
4. **Add query queue**: Implement Celery for async processing
5. **Database indexing**: Add indexes on frequently queried columns

## Summary

✅ **Increased resource limits** to match available hardware  
✅ **Added PostgreSQL performance tuning** for high-concurrency workloads  
✅ **Implemented connection pooling** to prevent exhaustion  
✅ **Extended timeouts** for complex operations  

**Expected Result**: Both concurrent queries should complete successfully without timeouts or network errors.

