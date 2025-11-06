-- schema.sql
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- trigram (lexical)
CREATE EXTENSION IF NOT EXISTS unaccent;    -- optional, improves lex recall

-- Documents and chunks
CREATE TABLE documents (
  doc_id      UUID PRIMARY KEY,
  title       TEXT,
  source_path TEXT,
  meta        JSONB DEFAULT '{}',
  created_at  TIMESTAMP DEFAULT now()
);

-- BGE-M3 dense = 1024 dims. We'll normalize vectors client-side → cosine via <=>.
CREATE TABLE chunks (
  chunk_id    UUID PRIMARY KEY,
  doc_id      UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
  page_start  INT,
  page_end    INT,
  section     TEXT,
  text        TEXT NOT NULL,
  is_ocr      BOOLEAN DEFAULT FALSE,
  is_figure   BOOLEAN DEFAULT FALSE,
  -- lexical & vector fields
  lex         tsvector,
  emb         vector(1024),
  meta        JSONB DEFAULT '{}'
);

-- Indexes: hybrid (lexical + ANN). Choose HNSW for low latency.
CREATE INDEX idx_chunks_lex ON chunks USING GIN (lex);
CREATE INDEX idx_chunks_emb_hnsw ON chunks USING hnsw (emb vector_cosine_ops);

-- Helpful function to (re)build lex tsvector
-- (You can also compute lex in app-side code.)
