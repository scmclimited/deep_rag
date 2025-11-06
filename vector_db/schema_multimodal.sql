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
  meta        JSONB DEFAULT '{}',
  created_at  TIMESTAMP DEFAULT now()
);

-- CLIP-ViT-B/32 produces 512-dimensional embeddings (multi-modal)
-- Supports: text, images, and text+image combinations in same semantic space
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
  emb         vector(512),           -- CLIP embeddings (512 dims)
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

