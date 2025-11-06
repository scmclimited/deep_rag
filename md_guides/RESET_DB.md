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

1. **Initialize the database** with the new multi-modal schema from `schema_multimodal.sql`
2. **Create tables** with:
   - `emb` column as `vector(512)` (CLIP embeddings)
   - `content_type` column for multi-modal support
   - `image_path` column for image chunks
3. **Set up indexes** for hybrid retrieval (lexical + vector)
4. **Create triggers** for automatic lex vector updates

## Migration Script

The `migration_add_multimodal.sql` script is included in the docker-entrypoint-initdb.d folder for:
- **Testing future migrations** against existing data
- **Reference** for how to migrate from old schema to new schema
- **Manual migrations** if needed later

**Note**: The migration script will only run if the old schema exists. On a fresh start, only `schema_multimodal.sql` will be applied.

## Verifying Fresh Start

After starting fresh, verify the schema:

```bash
# Connect to database
docker compose exec db psql -U ${DB_USER} -d ${DB_NAME}

# Check table structure
\d chunks

# Should show:
# - emb vector(512)  (not 1024)
# - content_type text
# - image_path text
```

## Schema Files

- **`schema_multimodal.sql`**: Fresh schema with CLIP embeddings (512 dims)
- **`migration_add_multimodal.sql`**: Migration script for existing databases
- **`ingestion_schema.sql`**: Old schema (kept for reference)

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

