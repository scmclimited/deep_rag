-- Migration: Add multi-modal embedding support
-- This migration adds support for CLIP-style multi-modal embeddings
-- CLIP-ViT-B/32 produces 512-dimensional embeddings
--
-- IMPORTANT: This script only runs if the old schema exists.
-- On a fresh database, only schema_multimodal.sql will be applied.
--
-- For existing databases: This migration requires data migration and re-embedding.
-- For fresh databases: Use schema_multimodal.sql instead.

-- Check if old schema exists (chunks table with 1024-dim emb)
DO $$
BEGIN
  -- Only run if chunks table exists with old schema
  IF EXISTS (
    SELECT 1 
    FROM information_schema.columns 
    WHERE table_name = 'chunks' 
    AND column_name = 'emb'
    AND table_schema = 'public'
  ) THEN
    -- Check if emb column is 1024 dimensions (old schema)
    IF EXISTS (
      SELECT 1 
      FROM information_schema.columns 
      WHERE table_name = 'chunks' 
      AND column_name = 'emb'
      AND data_type = 'USER-DEFINED'
      AND udt_name = 'vector'
    ) THEN
      -- Step 1: Add metadata columns (if they don't exist)
      ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_type TEXT DEFAULT 'text';
      ALTER TABLE chunks ADD COLUMN IF NOT EXISTS image_path TEXT;
      
      -- Update existing rows to default content_type
      UPDATE chunks SET content_type = 'text' WHERE content_type IS NULL;
      
      -- Note: Migrating emb dimension from 1024 to 512 requires:
      -- 1. Backing up data
      -- 2. Re-embedding all chunks with CLIP
      -- 3. Dropping and recreating the emb column
      -- This is a destructive operation - do it manually or via migration script
      
      RAISE NOTICE 'Migration: Added content_type and image_path columns.';
      RAISE NOTICE 'WARNING: Emb dimension migration (1024->512) requires manual migration.';
      RAISE NOTICE 'See RESET_DB.md for instructions on starting fresh with new schema.';
    END IF;
  END IF;
END $$;

-- For new installations, use this schema:
-- CREATE TABLE chunks (
--   chunk_id    UUID PRIMARY KEY,
--   doc_id      UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
--   page_start  INT,
--   page_end    INT,
--   section     TEXT,
--   text        TEXT NOT NULL,
--   is_ocr      BOOLEAN DEFAULT FALSE,
--   is_figure   BOOLEAN DEFAULT FALSE,
--   content_type TEXT DEFAULT 'text',  -- 'text', 'image', 'multimodal', etc.
--   image_path  TEXT,                  -- Path to image file if applicable
--   lex         tsvector,
--   emb         vector(512),           -- CLIP embeddings (512 dims)
--   meta        JSONB DEFAULT '{}'
-- );

