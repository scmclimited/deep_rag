-- schema_multimodal.sql
-- Updated schema for multi-modal CLIP embeddings
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- trigram (lexical)
CREATE EXTENSION IF NOT EXISTS unaccent;    -- optional, improves lex recall

-- Documents and chunks
CREATE TABLE documents (
  doc_id      UUID PRIMARY KEY,
  title       TEXT,
  source_path TEXT,
  content_hash TEXT,              -- SHA256 hash of document content for duplicate detection
  meta        JSONB DEFAULT '{}',
  created_at  TIMESTAMP DEFAULT now()
);

-- Index for content_hash for faster duplicate lookups
CREATE INDEX idx_documents_content_hash ON documents(content_hash);

COMMENT ON COLUMN documents.content_hash IS 'SHA256 hash of document content for duplicate detection';

-- CLIP-ViT-L/14 produces 768-dimensional embeddings (multi-modal, upgraded from ViT-B/32)
-- Supports: text, images, and text+image combinations in same semantic space
-- Higher dimensions = better semantic representation and retrieval quality
-- Note: Can use ViT-B/32 (512 dims) for faster performance by setting EMBEDDING_DIM=512
CREATE TABLE chunks (
  chunk_id    UUID PRIMARY KEY,
  doc_id      UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
  page_start  INT,
  page_end    INT,
  section     TEXT,
  text        TEXT NOT NULL,
  is_ocr      BOOLEAN DEFAULT FALSE,
  is_figure   BOOLEAN DEFAULT FALSE,
  content_type TEXT DEFAULT 'text',  -- 'text', 'image', 'multimodal', 'pdf_text', 'pdf_image'
  image_path  TEXT,                  -- Path to image file if applicable
  -- lexical & vector fields
  lex         tsvector,
  emb         vector(768),           -- CLIP embeddings (768 dims for ViT-L/14, 512 for ViT-B/32)
  meta        JSONB DEFAULT '{}'
);

-- Indexes: hybrid (lexical + ANN). Choose HNSW for low latency.
CREATE INDEX idx_chunks_lex ON chunks USING GIN (lex);
CREATE INDEX idx_chunks_emb_hnsw ON chunks USING hnsw (emb vector_cosine_ops);
CREATE INDEX idx_chunks_content_type ON chunks (content_type);

-- Helpful function to (re)build lex tsvector
CREATE OR REPLACE FUNCTION update_chunk_lex()
RETURNS TRIGGER AS $$
BEGIN
  NEW.lex := to_tsvector('simple', unaccent(NEW.text));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_chunk_lex_trigger
  BEFORE INSERT OR UPDATE OF text ON chunks
  FOR EACH ROW
  EXECUTE FUNCTION update_chunk_lex();

-- Thread tracking table for audit and analysis
-- Tracks user interactions, thread sessions, and document retrievals
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

