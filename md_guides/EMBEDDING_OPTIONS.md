# Embedding Model Options

## Current Implementation

### Multi-Modal Embeddings: CLIP-ViT-L/14 (Recommended)

Deep RAG currently uses **CLIP-ViT-L/14** (`sentence-transformers/clip-ViT-L-14`) for multi-modal embeddings, providing a unified vector space for both text and images.

- **Model**: `sentence-transformers/clip-ViT-L-14`
- **Type**: Multi-modal (text + images in same embedding space)
- **Dimensions**: **768** (recommended for production)
- **Runs**: Locally via `sentence-transformers`
- **Use Case**: Production RAG applications with multi-modal support
- **Advantages**:
  - ✅ True multi-modal (text + images in same embedding space)
  - ✅ Can search images by text descriptions
  - ✅ Can search text by image similarity
  - ✅ No API dependencies (runs locally)
  - ✅ Better semantic representation (768 dims vs 512 dims)
  - ✅ Production-ready performance
- **Limitations**:
  - ❌ Larger model size (~400MB vs ~150MB for ViT-B/32)
  - ❌ Slightly slower than ViT-B/32 (but better quality)

### Legacy Option: CLIP-ViT-B/32 (Faster, Lower Quality)

For development or resource-constrained environments, you can use **CLIP-ViT-B/32** (`sentence-transformers/clip-ViT-B-32`):

- **Model**: `sentence-transformers/clip-ViT-B-32`
- **Type**: Multi-modal (text + images in same embedding space)
- **Dimensions**: **512** (legacy, faster)
- **Runs**: Locally via `sentence-transformers`
- **Use Case**: Development, resource-constrained environments
- **Advantages**:
  - ✅ Faster inference
  - ✅ Lower memory usage
  - ✅ Smaller model size (~150MB)
- **Limitations**:
  - ❌ Lower quality semantic representation (512 dims vs 768 dims)
  - ❌ Not recommended for production

### Reranker: BAAI/bge-reranker-base

- **Model**: Cross-encoder for reranking
- **Type**: Text-only
- **Use Case**: Reranking retrieved chunks for better precision
- **Runs**: Locally via `sentence-transformers`

## Model Comparison

| Property | CLIP-ViT-L/14 (Current) | CLIP-ViT-B/32 (Legacy) |
|----------|-------------------------|------------------------|
| **Embedding Dimensions** | **768** | 512 |
| **Max Token Length** | 77 tokens | 77 tokens |
| **Performance** | **Better semantic representation** | Faster, lower memory |
| **Model Size** | ~400MB | ~150MB |
| **Use Case** | Production, high-quality retrieval | Development, resource-constrained |
| **Multi-Modal** | ✅ Yes | ✅ Yes |
| **Local Execution** | ✅ Yes | ✅ Yes |

## Why CLIP-ViT-L/14?

1. **Higher Dimensions (768 vs 512)**: More dimensional space = better semantic representation and retrieval accuracy
2. **Multi-Modal**: Embeds text and images in the same vector space, enabling true multi-modal search
3. **Open-Source & Local**: Runs entirely locally without API dependencies
4. **pgvector Compatible**: 768 dimensions well within pgvector's 2,000 dimension limit
5. **Production Ready**: Better performance for real-world RAG applications

## Configuration

Set via environment variables in `.env`:

```bash
# Use CLIP-ViT-L/14 (768 dims, recommended for production)
CLIP_MODEL=sentence-transformers/clip-ViT-L-14
EMBEDDING_DIM=768

# Or use CLIP-ViT-B/32 (512 dims, faster, legacy)
# CLIP_MODEL=sentence-transformers/clip-ViT-B-32
# EMBEDDING_DIM=512
```

## Migration

### Upgrading from ViT-B/32 to ViT-L/14

If you have an existing database with ViT-B/32 (512 dims), see `vector_db/migration_upgrade_to_768.sql` for migration steps.

**Migration Steps**:
1. Backup your database
2. Run the migration script:
   ```bash
   docker compose exec db psql -U $DB_USER -d $DB_NAME -f /docker-entrypoint-initdb.d/migration_upgrade_to_768.sql
   ```
3. Update `.env`:
   ```bash
   CLIP_MODEL=sentence-transformers/clip-ViT-L-14
   EMBEDDING_DIM=768
   ```
4. Re-ingest documents (old 512-dim embeddings won't work with 768-dim model)

### Downgrading from ViT-L/14 to ViT-B/32

If you need to downgrade (not recommended):

1. Backup your database
2. Update `.env`:
   ```bash
   CLIP_MODEL=sentence-transformers/clip-ViT-B-32
   EMBEDDING_DIM=512
   ```
3. Re-ingest documents (old 768-dim embeddings won't work with 512-dim model)

## Token Limit Handling

CLIP models have a 77 token limit (inherent to architecture). Deep RAG handles this through:
- **Intelligent Chunking**: Chunks are limited to ~25 words (≈32-37 tokens) with safe margin
- **Token-Aware Truncation**: Uses CLIP's tokenizer for accurate truncation
- **Fallback Handling**: Conservative word-based truncation if tokenizer unavailable
- **Retry Logic**: Multiple truncation attempts with progressively smaller chunks

## Installation

```bash
# Already installed via requirements.txt
pip install sentence-transformers

# No additional installation needed!
# sentence-transformers includes CLIP models
```

## Summary

- **Current**: CLIP-ViT-L/14 (768 dims) is recommended for production ✅
- **Legacy**: CLIP-ViT-B/32 (512 dims) available for development/resource-constrained environments
- **Multi-Modal**: Both models support text + images in unified embedding space ✅
- **No API dependency**: All models run locally ✅
- **Easy upgrade**: Migration scripts available for upgrading from 512 to 768 dims

## Recommendation

**For Production**: Use CLIP-ViT-L/14 (768 dims) for better semantic representation and retrieval quality.

**For Development**: Use CLIP-ViT-B/32 (512 dims) if memory/speed is critical, but expect lower quality results.
