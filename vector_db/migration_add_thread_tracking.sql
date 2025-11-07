-- migration_add_thread_tracking.sql
-- Migration script to add thread tracking table for audit purposes
-- 
-- This table tracks user interactions, thread sessions, and document retrievals
-- to enable auditing and analysis of user behavior and system outputs

CREATE TABLE IF NOT EXISTS thread_tracking (
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

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_thread_tracking_user_id ON thread_tracking(user_id);
CREATE INDEX IF NOT EXISTS idx_thread_tracking_thread_id ON thread_tracking(thread_id);
CREATE INDEX IF NOT EXISTS idx_thread_tracking_doc_ids ON thread_tracking USING GIN(doc_ids);
CREATE INDEX IF NOT EXISTS idx_thread_tracking_created_at ON thread_tracking(created_at);
CREATE INDEX IF NOT EXISTS idx_thread_tracking_entry_point ON thread_tracking(entry_point);
CREATE INDEX IF NOT EXISTS idx_thread_tracking_pipeline_type ON thread_tracking(pipeline_type);

-- Composite index for common queries (user + thread)
CREATE INDEX IF NOT EXISTS idx_thread_tracking_user_thread ON thread_tracking(user_id, thread_id);

COMMENT ON TABLE thread_tracking IS 'Tracks user interactions, thread sessions, and document retrievals for audit and analysis';
COMMENT ON COLUMN thread_tracking.user_id IS 'User identifier from external authentication system';
COMMENT ON COLUMN thread_tracking.thread_id IS 'Thread/session identifier for conversation tracking';
COMMENT ON COLUMN thread_tracking.doc_ids IS 'Array of document IDs retrieved during this interaction';
COMMENT ON COLUMN thread_tracking.graphstate IS 'Full graph state metadata including all agent steps (planner, retriever, compressor, critic, synthesizer)';
COMMENT ON COLUMN thread_tracking.ingestion_meta IS 'Metadata from ingestion operations (doc_id, title, chunk_count, etc.)';
COMMENT ON COLUMN thread_tracking.entry_point IS 'Entry point used: cli, rest, make, toml';
COMMENT ON COLUMN thread_tracking.pipeline_type IS 'Pipeline type used: direct (inference/agents/pipeline.py) or langgraph';
COMMENT ON COLUMN thread_tracking.cross_doc IS 'Whether cross-document retrieval was enabled for this interaction';

