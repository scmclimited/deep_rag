-- migration_upgrade_to_768.sql
-- Migration script to upgrade from CLIP-ViT-B/32 (512 dims) to CLIP-ViT-L/14 (768 dims)
-- 
-- WARNING: This migration requires re-embedding all chunks with the new model
-- The old 512-dim embeddings are NOT compatible with 768-dim vector space
-- 
-- BACKUP YOUR DATABASE BEFORE RUNNING THIS MIGRATION
--
-- Steps:
-- 1. Backup existing data
-- 2. Drop and recreate the embedding column with new dimensions
-- 3. Drop and recreate the HNSW index
-- 4. Re-ingest all documents with new embedding model

-- Step 1: Create backup table (optional but recommended)
CREATE TABLE IF NOT EXISTS chunks_backup_512 AS 
SELECT * FROM chunks;

-- Step 2: Drop existing vector index (must be done before altering column)
DROP INDEX IF EXISTS idx_chunks_emb_hnsw;

-- Step 3: Drop the old embedding column
ALTER TABLE chunks DROP COLUMN IF EXISTS emb;

-- Step 4: Add new embedding column with 768 dimensions
ALTER TABLE chunks ADD COLUMN emb vector(768);

-- Step 5: Recreate the HNSW index
CREATE INDEX idx_chunks_emb_hnsw ON chunks USING hnsw (emb vector_cosine_ops);

-- Step 6: Verify the schema
\d chunks

-- NOTE: After running this migration, you MUST re-ingest all documents
-- using the new CLIP-ViT-L/14 model (768 dims) to populate the embeddings.
--
-- To re-ingest:
--   1. Set environment variables:
--      export CLIP_MODEL="sentence-transformers/clip-ViT-L-14"
--      export EMBEDDING_DIM="768"
--   2. Re-run ingestion for all documents
--
-- To rollback (if needed):
--   1. Drop chunks table: DROP TABLE chunks;
--   2. Recreate from backup: ALTER TABLE chunks_backup_512 RENAME TO chunks;
--   3. Recreate index: CREATE INDEX idx_chunks_emb_hnsw ON chunks USING hnsw (emb vector_cosine_ops);

