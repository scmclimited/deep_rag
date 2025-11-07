# Thread Tracking and Audit Logging

This guide explains the thread tracking and audit logging system in Deep RAG, including table schemas, use cases, and how to use it for SFT/RLHF training.

---

## Overview

Deep RAG includes comprehensive audit logging via the `thread_tracking` table. This system tracks:
- **User interactions**: Who is using the system
- **Thread sessions**: Conversation state across multiple queries
- **Document retrievals**: Which documents were retrieved for each query
- **Pipeline states**: Full agentic reasoning steps (planner, retriever, compressor, critic, synthesizer)
- **Entry points**: How users are accessing the system (CLI, REST, Make, TOML)
- **Cross-document retrieval**: Whether `--cross-doc` flag was used

This data is essential for:
- **Supervised Fine-Tuning (SFT)**: Training models on high-quality query-answer pairs
- **Reinforcement Learning from Human Feedback (RLHF)**: Learning from user interactions
- **Audit and Compliance**: Tracking system usage and behavior
- **Performance Analysis**: Understanding retrieval quality and user behavior

---

## Database Schema

### `thread_tracking` Table

```sql
CREATE TABLE thread_tracking (
  id              SERIAL PRIMARY KEY,
  user_id         TEXT NOT NULL,                    -- User identifier (from external auth system)
  thread_id       TEXT NOT NULL,                    -- Thread/session identifier
  doc_ids         TEXT[] DEFAULT '{}',             -- Array of document IDs retrieved in this thread
  query_text      TEXT,                             -- Original query/question
  final_answer    TEXT,                             -- Final synthesized answer
  graphstate      JSONB DEFAULT '{}',              -- Full graph state metadata (all agent steps)
  ingestion_meta  JSONB DEFAULT '{}',              -- Ingestion metadata (if ingestion occurred)
  created_at      TIMESTAMP DEFAULT now(),         -- When the interaction started
  completed_at    TIMESTAMP,                        -- When the interaction completed
  entry_point     TEXT,                             -- Entry point: 'cli', 'rest', 'make', 'toml'
  pipeline_type   TEXT,                             -- Pipeline: 'direct', 'langgraph'
  cross_doc       BOOLEAN DEFAULT FALSE,           -- Whether cross-document retrieval was enabled
  metadata        JSONB DEFAULT '{}'                -- Additional metadata
);
```

### Indexes

```sql
CREATE INDEX idx_thread_tracking_user_id ON thread_tracking(user_id);
CREATE INDEX idx_thread_tracking_thread_id ON thread_tracking(thread_id);
CREATE INDEX idx_thread_tracking_doc_ids ON thread_tracking USING GIN(doc_ids);
CREATE INDEX idx_thread_tracking_created_at ON thread_tracking(created_at);
CREATE INDEX idx_thread_tracking_entry_point ON thread_tracking(entry_point);
CREATE INDEX idx_thread_tracking_pipeline_type ON thread_tracking(pipeline_type);
CREATE INDEX idx_thread_tracking_user_thread ON thread_tracking(user_id, thread_id);
```

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key, auto-incrementing |
| `user_id` | TEXT | User identifier from external authentication system |
| `thread_id` | TEXT | Thread/session identifier for conversation tracking |
| `doc_ids` | TEXT[] | Array of document IDs retrieved during this interaction |
| `query_text` | TEXT | Original query/question |
| `final_answer` | TEXT | Final synthesized answer |
| `graphstate` | JSONB | Full graph state metadata including all agent steps (planner, retriever, compressor, critic, synthesizer) |
| `ingestion_meta` | JSONB | Metadata from ingestion operations (doc_id, title, chunk_count, etc.) |
| `created_at` | TIMESTAMP | When the interaction started |
| `completed_at` | TIMESTAMP | When the interaction completed |
| `entry_point` | TEXT | Entry point used: 'cli', 'rest', 'make', 'toml' |
| `pipeline_type` | TEXT | Pipeline type used: 'direct' (inference/agents/pipeline.py) or 'langgraph' |
| `cross_doc` | BOOLEAN | Whether cross-document retrieval was enabled |
| `metadata` | JSONB | Additional metadata (scores, timings, etc.) |

---

## Usage

### Logging Thread Interactions

Thread interactions are automatically logged when using the LangGraph pipeline (`query-graph`, `infer-graph`). For the direct pipeline, you can manually log interactions using the `retrieval.thread_tracking` module.

#### Automatic Logging (LangGraph Pipeline)

The LangGraph pipeline automatically logs interactions to the `thread_tracking` table. No additional code is required.

#### Manual Logging (Direct Pipeline)

```python
from retrieval.thread_tracking import log_thread_interaction

# Log a thread interaction
record_id = log_thread_interaction(
    user_id="user_123",
    thread_id="session_1",
    query_text="What are the requirements?",
    doc_ids=["550e8400-e29b-41d4-a716-446655440000"],
    final_answer="The requirements are...",
    entry_point="cli",
    pipeline_type="direct",
    cross_doc=False,
    metadata={"confidence": 0.85, "iterations": 1}
)
```

### Retrieving Thread Interactions

```python
from retrieval.thread_tracking import get_thread_interactions

# Get all interactions for a user
interactions = get_thread_interactions(user_id="user_123", limit=100)

# Get all interactions for a thread
interactions = get_thread_interactions(thread_id="session_1", limit=100)

# Get all interactions (no filter)
interactions = get_thread_interactions(limit=100)
```

### Updating Thread Interactions

```python
from retrieval.thread_tracking import update_thread_interaction

# Update a thread interaction
success = update_thread_interaction(
    record_id=1,
    final_answer="Updated answer...",
    doc_ids=["550e8400-e29b-41d4-a716-446655440000", "660e8400-e29b-41d4-a716-446655440001"],
    metadata={"confidence": 0.90, "iterations": 2}
)
```

---

## Use Cases

### Use Case 1: Supervised Fine-Tuning (SFT)

**Goal**: Train a model on high-quality query-answer pairs from user interactions.

**Workflow**:
1. Extract query-answer pairs from `thread_tracking` table
2. Filter for high-quality interactions (high confidence, user feedback, etc.)
3. Format data for SFT training
4. Train model on extracted data

**Example**:
```python
import psycopg2
import pandas as pd
from retrieval.db_utils import connect

# Connect to database
with connect() as conn:
    # Query high-quality interactions
    df = pd.read_sql("""
        SELECT 
            query_text,
            final_answer,
            graphstate,
            metadata,
            cross_doc,
            pipeline_type
        FROM thread_tracking
        WHERE final_answer IS NOT NULL
          AND graphstate != '{}'
          AND (metadata->>'confidence')::float > 0.7
        ORDER BY created_at DESC
        LIMIT 1000
    """, conn)
    
    # Extract query-answer pairs
    training_data = []
    for _, row in df.iterrows():
        training_data.append({
            "question": row["query_text"],
            "answer": row["final_answer"],
            "context": row["graphstate"],
            "metadata": row["metadata"]
        })
    
    # Format for SFT training
    # ... (format data for your training framework)
```

---

### Use Case 2: Reinforcement Learning from Human Feedback (RLHF)

**Goal**: Learn from user interactions and feedback to improve model performance.

**Workflow**:
1. Track user interactions and feedback
2. Analyze which interactions received positive/negative feedback
3. Use feedback to improve retrieval and synthesis
4. Iterate on model based on feedback patterns

**Example**:
```python
# Add feedback to thread_tracking metadata
from retrieval.thread_tracking import update_thread_interaction

# Update interaction with user feedback
update_thread_interaction(
    record_id=interaction_id,
    metadata={
        "confidence": 0.85,
        "iterations": 1,
        "user_feedback": "positive",  # or "negative"
        "feedback_score": 5,  # 1-5 scale
        "feedback_text": "Great answer, very helpful!"
    }
)

# Analyze feedback patterns
with connect() as conn:
    df = pd.read_sql("""
        SELECT 
            query_text,
            final_answer,
            metadata->>'user_feedback' as feedback,
            metadata->>'feedback_score' as score
        FROM thread_tracking
        WHERE metadata->>'user_feedback' IS NOT NULL
        ORDER BY created_at DESC
    """, conn)
    
    # Analyze positive vs negative feedback
    positive_feedback = df[df["feedback"] == "positive"]
    negative_feedback = df[df["feedback"] == "negative"]
    
    # Use feedback to improve model
    # ... (analyze patterns and improve retrieval/synthesis)
```

---

### Use Case 3: Audit and Compliance

**Goal**: Track system usage and behavior for audit and compliance purposes.

**Workflow**:
1. Query `thread_tracking` table for audit reports
2. Analyze user behavior and system usage
3. Generate compliance reports
4. Track document access and retrieval patterns

**Example**:
```python
# Generate audit report
with connect() as conn:
    df = pd.read_sql("""
        SELECT 
            user_id,
            thread_id,
            query_text,
            doc_ids,
            entry_point,
            pipeline_type,
            cross_doc,
            created_at,
            completed_at
        FROM thread_tracking
        WHERE created_at >= NOW() - INTERVAL '30 days'
        ORDER BY created_at DESC
    """, conn)
    
    # Generate audit report
    audit_report = {
        "total_interactions": len(df),
        "unique_users": df["user_id"].nunique(),
        "unique_threads": df["thread_id"].nunique(),
        "entry_points": df["entry_point"].value_counts().to_dict(),
        "pipeline_types": df["pipeline_type"].value_counts().to_dict(),
        "cross_doc_usage": df["cross_doc"].sum(),
        "doc_access_patterns": df["doc_ids"].explode().value_counts().to_dict()
    }
    
    # Save audit report
    # ... (save to file or database)
```

---

### Use Case 4: Performance Analysis

**Goal**: Understand retrieval quality and user behavior to improve system performance.

**Workflow**:
1. Analyze retrieval patterns and document access
2. Identify which queries require refinement
3. Track confidence scores and iterations
4. Optimize retrieval parameters based on analysis

**Example**:
```python
# Analyze retrieval performance
with connect() as conn:
    df = pd.read_sql("""
        SELECT 
            query_text,
            doc_ids,
            graphstate,
            metadata,
            cross_doc,
            pipeline_type
        FROM thread_tracking
        WHERE graphstate != '{}'
        ORDER BY created_at DESC
        LIMIT 1000
    """, conn)
    
    # Analyze retrieval patterns
    retrieval_analysis = {
        "avg_docs_per_query": df["doc_ids"].apply(len).mean(),
        "cross_doc_usage_rate": df["cross_doc"].mean(),
        "avg_confidence": df["metadata"].apply(lambda x: x.get("confidence", 0) if x else 0).mean(),
        "avg_iterations": df["metadata"].apply(lambda x: x.get("iterations", 0) if x else 0).mean(),
        "queries_requiring_refinement": len(df[df["metadata"].apply(lambda x: x.get("iterations", 0) if x else 0 > 1)])
    }
    
    # Use analysis to improve system
    # ... (optimize retrieval parameters, improve prompts, etc.)
```

---

## Migration

To add the `thread_tracking` table to your database, run the migration script:

```bash
# Run migration
docker compose exec db psql -U $DB_USER -d $DB_NAME -f /path/to/vector_db/migration_add_thread_tracking.sql

# Or manually
psql -U $DB_USER -d $DB_NAME -f vector_db/migration_add_thread_tracking.sql
```

The migration script:
- Creates the `thread_tracking` table
- Creates all necessary indexes
- Adds table and column comments

---

## Best Practices

1. **Always log interactions**: Use automatic logging (LangGraph) or manual logging (Direct pipeline)
2. **Include user_id**: Always provide a valid `user_id` from your authentication system
3. **Use thread_id for conversations**: Use consistent `thread_id` values for multi-turn conversations
4. **Store complete graphstate**: Include full agentic reasoning steps in `graphstate` for SFT training
5. **Add metadata**: Include confidence scores, iterations, and other relevant metadata
6. **Track entry points**: Log which entry point (CLI, REST, Make, TOML) was used
7. **Track cross_doc usage**: Log whether `--cross-doc` flag was used for analysis
8. **Regular cleanup**: Archive old interactions to maintain database performance

---

## Query Examples

### Get All Interactions for a User

```sql
SELECT * FROM thread_tracking
WHERE user_id = 'user_123'
ORDER BY created_at DESC;
```

### Get All Interactions for a Thread

```sql
SELECT * FROM thread_tracking
WHERE thread_id = 'session_1'
ORDER BY created_at DESC;
```

### Get High-Quality Interactions for SFT

```sql
SELECT 
    query_text,
    final_answer,
    graphstate,
    metadata
FROM thread_tracking
WHERE final_answer IS NOT NULL
  AND graphstate != '{}'
  AND (metadata->>'confidence')::float > 0.7
ORDER BY created_at DESC
LIMIT 1000;
```

### Get Interactions Using Cross-Document Retrieval

```sql
SELECT * FROM thread_tracking
WHERE cross_doc = true
ORDER BY created_at DESC;
```

### Get Interactions by Entry Point

```sql
SELECT 
    entry_point,
    COUNT(*) as count
FROM thread_tracking
GROUP BY entry_point
ORDER BY count DESC;
```

### Get Document Access Patterns

```sql
SELECT 
    unnest(doc_ids) as doc_id,
    COUNT(*) as access_count
FROM thread_tracking
WHERE doc_ids != '{}'
GROUP BY doc_id
ORDER BY access_count DESC;
```

---

## Integration with Entry Points

Thread tracking is automatically integrated with:
- **LangGraph pipeline**: Automatic logging for `query-graph` and `infer-graph`
- **Direct pipeline**: Manual logging available via `retrieval.thread_tracking` module
- **All entry points**: CLI, REST, Make, TOML all support thread tracking

For detailed entry point scenarios, see [`ENTRY_POINTS_AND_SCENARIOS.md`](ENTRY_POINTS_AND_SCENARIOS.md).

---

## Future Enhancements

Potential future enhancements to thread tracking:
- **User feedback integration**: Direct feedback collection in the UI
- **Real-time analytics**: Dashboard for monitoring interactions
- **Automated SFT dataset generation**: Automatic extraction and formatting for training
- **A/B testing support**: Track different pipeline configurations
- **Cost tracking**: Monitor LLM API costs per interaction

---

For more information on entry points and scenarios, see [`ENTRY_POINTS_AND_SCENARIOS.md`](ENTRY_POINTS_AND_SCENARIOS.md).

