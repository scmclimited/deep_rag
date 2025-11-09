# Suggested API Routes for Enhanced Frontend Experience

This document outlines additional API routes that would enhance the Streamlit frontend to provide a ChatGPT/Merlin AI-like experience.

## Current Routes (Already Implemented)

✅ `GET /health` - Health check  
✅ `POST /ingest` - Single file ingestion  
✅ `POST /ask` - Query (direct pipeline)  
✅ `POST /ask-graph` - Query (LangGraph pipeline) with thread_id  
✅ `POST /infer` - Ingest + query (direct pipeline)  
✅ `POST /infer-graph` - Ingest + query (LangGraph pipeline) with thread_id  
✅ `GET /diagnostics/document` - Document diagnostics  
✅ `GET /graph` - Graph export  

## Recommended New Routes

### Thread Management

#### 1. `GET /threads`
**Purpose**: List all threads for a user  
**Query Parameters**:
- `user_id` (optional): Filter by user ID
- `limit` (optional, default: 100): Maximum number of threads to return

**Response**:
```json
{
  "threads": [
    {
      "thread_id": "uuid",
      "title": "Thread title (first message or auto-generated)",
      "user_id": "user_id",
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z",
      "message_count": 10,
      "doc_ids": ["doc_id1", "doc_id2"]
    }
  ],
  "total": 50
}
```

**Implementation**: Query `thread_tracking` table grouped by `thread_id`

---

#### 2. `GET /threads/{thread_id}`
**Purpose**: Get thread details and message history  
**Path Parameters**:
- `thread_id`: Thread identifier

**Query Parameters**:
- `limit` (optional, default: 100): Maximum messages to return

**Response**:
```json
{
  "thread_id": "uuid",
  "title": "Thread title",
  "user_id": "user_id",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "What is...",
      "created_at": "2024-01-01T00:00:00Z",
      "doc_id": "doc_id",
      "attachment": "filename.pdf"
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "The answer is...",
      "created_at": "2024-01-01T00:00:01Z",
      "doc_id": "doc_id",
      "confidence": 0.95
    }
  ],
  "doc_ids": ["doc_id1", "doc_id2"]
}
```

**Implementation**: Query `thread_tracking` table filtered by `thread_id`, order by `created_at`

---

#### 3. `POST /threads`
**Purpose**: Create a new thread  
**Request Body**:
```json
{
  "title": "Optional thread title",
  "user_id": "user_id"
}
```

**Response**:
```json
{
  "thread_id": "uuid",
  "title": "Thread title",
  "created_at": "2024-01-01T00:00:00Z"
}
```

**Implementation**: Generate new `thread_id`, optionally create initial record in `thread_tracking`

---

#### 4. `DELETE /threads/{thread_id}`
**Purpose**: Delete a thread and its history  
**Path Parameters**:
- `thread_id`: Thread identifier

**Response**:
```json
{
  "status": "success",
  "thread_id": "uuid",
  "deleted_messages": 10
}
```

**Implementation**: Delete all records from `thread_tracking` where `thread_id` matches

---

#### 5. `PATCH /threads/{thread_id}`
**Purpose**: Update thread metadata (e.g., title)  
**Path Parameters**:
- `thread_id`: Thread identifier

**Request Body**:
```json
{
  "title": "New thread title"
}
```

**Response**:
```json
{
  "thread_id": "uuid",
  "title": "New thread title",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

**Implementation**: Update thread metadata (may require new `thread_metadata` table or JSONB field)

---

### Document Management

#### 6. `GET /documents`
**Purpose**: List all documents in the knowledge base  
**Query Parameters**:
- `limit` (optional, default: 100): Maximum documents to return
- `offset` (optional, default: 0): Pagination offset
- `search` (optional): Search by title

**Response**:
```json
{
  "documents": [
    {
      "doc_id": "uuid",
      "title": "Document title",
      "source_path": "/path/to/file.pdf",
      "created_at": "2024-01-01T00:00:00Z",
      "chunk_count": 150,
      "page_count": 25
    }
  ],
  "total": 50,
  "limit": 100,
  "offset": 0
}
```

**Implementation**: Query `documents` table, join with `chunks` for counts

---

#### 7. `GET /documents/{doc_id}`
**Purpose**: Get document details  
**Path Parameters**:
- `doc_id`: Document identifier

**Response**:
```json
{
  "doc_id": "uuid",
  "title": "Document title",
  "source_path": "/path/to/file.pdf",
  "created_at": "2024-01-01T00:00:00Z",
  "chunk_count": 150,
  "page_count": 25,
  "metadata": {},
  "chunks": [
    {
      "chunk_id": "uuid",
      "page_start": 1,
      "page_end": 1,
      "text_preview": "First 200 chars..."
    }
  ]
}
```

**Implementation**: Query `documents` and `chunks` tables

---

#### 8. `DELETE /documents/{doc_id}`
**Purpose**: Delete a document and all its chunks  
**Path Parameters**:
- `doc_id`: Document identifier

**Response**:
```json
{
  "status": "success",
  "doc_id": "uuid",
  "deleted_chunks": 150
}
```

**Implementation**: Delete from `documents` table (cascade deletes chunks)

---

### Batch Operations

#### 9. `POST /ingest/batch`
**Purpose**: Ingest multiple files in a single request  
**Request**: Multipart form data
- `files`: Array of file uploads
- `titles`: Optional array of titles (one per file)

**Response**:
```json
{
  "status": "success",
  "results": [
    {
      "filename": "file1.pdf",
      "doc_id": "uuid",
      "status": "success",
      "chunk_count": 50
    },
    {
      "filename": "file2.pdf",
      "status": "error",
      "error": "Unsupported file type"
    }
  ],
  "total": 2,
  "successful": 1,
  "failed": 1
}
```

**Implementation**: Process files sequentially or in parallel, return aggregated results

---

### Streaming Support

#### 10. `GET /ask-graph/stream` or WebSocket `/ws/ask-graph`
**Purpose**: Stream responses in real-time  
**Request**: Same as `POST /ask-graph` but via WebSocket or SSE

**Response**: Server-Sent Events (SSE) or WebSocket messages:
```
data: {"type": "token", "content": "The"}
data: {"type": "token", "content": " answer"}
data: {"type": "complete", "answer": "The answer is..."}
```

**Implementation**: Modify LangGraph pipeline to yield tokens/partial responses

---

#### 11. `GET /infer-graph/stream` or WebSocket `/ws/infer-graph`
**Purpose**: Stream responses for ingest + query operations  
**Similar to above but includes ingestion progress**

**Response**:
```
data: {"type": "ingestion", "status": "processing", "filename": "file.pdf"}
data: {"type": "ingestion", "status": "complete", "doc_id": "uuid"}
data: {"type": "token", "content": "The"}
...
```

---

### Enhanced Querying

#### 12. `POST /ask-graph/context`
**Purpose**: Query with additional context (e.g., previous messages)  
**Request Body**:
```json
{
  "question": "What is...",
  "thread_id": "uuid",
  "context": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ],
  "doc_id": "uuid",
  "cross_doc": false
}
```

**Implementation**: Include context in planner agent prompt

---

## Implementation Priority

### High Priority (Essential for ChatGPT-like experience):
1. ✅ `GET /threads` - Thread listing
2. ✅ `GET /threads/{thread_id}` - Thread history
3. ✅ `GET /documents` - Document listing
4. ✅ `POST /ingest/batch` - Batch ingestion

### Medium Priority (Enhanced UX):
5. `POST /threads` - Create thread with metadata
6. `DELETE /threads/{thread_id}` - Delete threads
7. `GET /documents/{doc_id}` - Document details
8. `DELETE /documents/{doc_id}` - Document deletion

### Low Priority (Nice to have):
9. `PATCH /threads/{thread_id}` - Update thread metadata
10. Streaming endpoints (WebSocket/SSE)
11. `POST /ask-graph/context` - Context-aware queries

## LangGraph Chain Considerations

### Current State
The existing LangGraph chain already supports:
- ✅ `thread_id` for conversation state
- ✅ Conditional routing and refinement
- ✅ Thread tracking via `thread_tracking` table

### Recommendations

**Option 1: Enhance Existing Chain (Recommended)**
- The current LangGraph chain is sufficient for thread management
- Add context injection node to include previous messages
- No new chain needed, just enhance existing nodes

**Option 2: Create Thread-Aware Chain**
- Create a new chain specifically for UI interactions
- Include nodes for:
  - Context retrieval (previous messages)
  - Thread state management
  - Response formatting for UI
- This would be a wrapper around the existing chain

**Recommendation**: **Option 1** - Enhance the existing chain with a context injection node. The current architecture is already thread-aware, so we just need to:
1. Add a node that retrieves previous messages from `thread_tracking`
2. Inject context into the planner node
3. Optionally add a response formatting node for UI

## Database Considerations

### New Tables (Optional)

**`thread_metadata`** (if we want richer thread management):
```sql
CREATE TABLE thread_metadata (
  thread_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT,
  created_at TIMESTAMP DEFAULT now(),
  updated_at TIMESTAMP DEFAULT now(),
  metadata JSONB DEFAULT '{}'
);
```

**Note**: This is optional - we can derive thread metadata from `thread_tracking` table.

## Example Implementation

See `deep_rag/deep_rag_backend/inference/routes/` for reference implementations. New routes should follow the same pattern:
- Use FastAPI routers
- Include proper error handling
- Log operations
- Return consistent JSON responses
- Support query parameters for filtering/pagination

