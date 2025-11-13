# Resetting the Database to Start Fresh

## Quick Start: Reset Database to Fresh State

If you want to start fresh with the new multi-modal schema, follow these steps:

### Option 1: Remove Existing Containers and Volumes (Recommended)

```bash
# Stop and remove containers
docker compose down

# Remove the database volume (if it exists)
docker volume rm deep_rag_db_data 2>/dev/null || true

# Remove any orphaned volumes
docker volume prune -f

# Start fresh with new schema
docker compose up -d --build
```

### Option 2: Force Recreate Database Container

```bash
# Stop and remove containers (including volumes)
docker compose down -v

# Start fresh
docker compose up -d --build
```

### Option 3: Manual Database Reset (if using named volume)

```bash
# Stop containers
docker compose down

# Remove specific volume
docker volume rm deep_rag_db_data

# Start fresh
docker compose up -d
```

## What Happens on Fresh Start

When you start fresh, PostgreSQL will:

1. **Initialize the database** with the complete multi-modal schema from `schema_multimodal.sql`
2. **Create tables** with:
   - `emb` column as `vector(768)` (openai/clip-vit-large-patch14-336 embeddings)
   - `content_type` column for multi-modal support
   - `image_path` column for image chunks
   - `content_hash` column for duplicate detection
   - `thread_tracking` table for conversation history
3. **Set up indexes** for hybrid retrieval (lexical + vector)
4. **Create triggers** for automatic lex vector updates

## Schema Initialization

On a fresh start, `schema_multimodal.sql` is automatically applied when the database container starts. This schema includes:
- Multi-modal CLIP embeddings (768 dims)
- Content hash for duplicate detection
- Thread tracking for conversation history
- All necessary indexes and triggers

## Verifying Fresh Start

After starting fresh, verify the schema:

```bash
# Connect to database
docker compose exec db psql -U ${DB_USER} -d ${DB_NAME}

# Check table structure
\d chunks

# Should show:
# - emb vector(768)  (openai/clip-vit-large-patch14-336)
# - content_type text
# - image_path text
# - content_hash text (in documents table)
```

## Schema Files

- **`schema_multimodal.sql`**: Complete schema with CLIP embeddings (768 dims), content_hash, thread_tracking, and multi-modal support

## Troubleshooting

If you see errors about existing tables or columns:

1. **Ensure containers are fully stopped**:
   ```bash
   docker compose down -v
   ```

2. **Check for orphaned volumes**:
   ```bash
   docker volume ls | grep deep_rag
   ```

3. **Remove all volumes**:
   ```bash
   docker volume rm $(docker volume ls -q | grep deep_rag) 2>/dev/null || true
   ```

4. **Start completely fresh**:
   ```bash
   docker compose up -d --force-recreate
   ```

