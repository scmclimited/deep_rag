# Embedding Model Options

## Current Implementation

### Text Embeddings: `BAAI/bge-m3`
- **Model**: `BAAI/bge-m3` (Beijing Academy of Artificial Intelligence)
- **Type**: Text-only dense embeddings
- **Dimensions**: 1024
- **Runs**: Locally via `sentence-transformers`
- **Use Case**: Excellent for text-based RAG
- **Advantages**:
  - ✅ No API dependencies (runs locally)
  - ✅ Fast inference
  - ✅ Good multilingual support
  - ✅ High quality for text retrieval
- **Limitations**:
  - ❌ Text-only (cannot embed images directly)
  - ❌ For images, we extract text via OCR only

### Reranker: `BAAI/bge-reranker-base`
- **Model**: Cross-encoder for reranking
- **Type**: Text-only
- **Use Case**: Reranking retrieved chunks for better precision

## Current Image Handling

For images (PNG, JPEG), we currently:
1. Extract text via OCR (Tesseract)
2. Embed the extracted text using `BAAI/bge-m3`
3. Store text embeddings (not visual embeddings)

**Limitation**: Visual content without text is not searchable.

## Future: CLIP-Style Multi-Modal Embeddings

### Option 1: CLIP (OpenAI)
- **Model**: `openai/clip-vit-base-patch32` or larger variants
- **Type**: Vision-language model (multi-modal)
- **Runs**: Locally via `transformers` or `timm`
- **Advantages**:
  - ✅ True multi-modal (text + images in same embedding space)
  - ✅ Can search images by text descriptions
  - ✅ Can search text by image similarity
  - ✅ Runs locally
- **Requirements**:
  - Additional storage for image embeddings
  - May need separate embedding column for images
  - Larger model size

### Option 2: OpenCLIP (Meta)
- **Model**: `laion/CLIP-ViT-B-32` or similar
- **Type**: Open-source CLIP variant
- **Runs**: Locally via `open_clip`
- **Advantages**:
  - ✅ Open-source alternative to CLIP
  - ✅ Multiple model sizes available
  - ✅ Good performance
- **Install**: `pip install open-clip-torch`

### Option 3: BLIP-2 or BLIP
- **Model**: `Salesforce/blip2-opt-2.7b` or `Salesforce/blip-image-captioning-base`
- **Type**: Vision-language model with captioning
- **Use Case**: Generate text descriptions from images, then embed text
- **Advantages**:
  - ✅ Generates rich captions for images
  - ✅ Can use existing text embedding model on captions
- **Limitations**:
  - Two-step process (caption → embed)
  - Slower than direct embeddings

### Option 4: Sentence Transformers Multi-Modal
- **Model**: `sentence-transformers/clip-ViT-B-32`
- **Type**: Multi-modal via sentence-transformers
- **Runs**: Locally via `sentence-transformers`
- **Advantages**:
  - ✅ Easy integration with existing codebase
  - ✅ Same API as current `BAAI/bge-m3`
  - ✅ Unified interface for text and images

## Recommendation

### Current Setup (Text-Only)
**Keep `BAAI/bge-m3`** - It's excellent for text-based RAG and runs locally without API dependencies.

### For Multi-Modal Support
**Add `sentence-transformers/clip-ViT-B-32`** as an optional image embedding model:

1. **Dual Embedding Strategy**:
   - Text chunks: Use `BAAI/bge-m3` (current)
   - Image chunks: Use `clip-ViT-B-32` (new)
   - Store both in database (separate columns or unified with type indicator)

2. **Implementation**:
   ```python
   # In ingestion/ingest_image.py
   from sentence_transformers import SentenceTransformer
   
   # For images
   clip_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')
   image_embedding = clip_model.encode(image)
   
   # For text from images (OCR)
   text_embedding = emb_model.encode(ocr_text)  # Current BAAI/bge-m3
   ```

3. **Database Schema** (optional enhancement):
   ```sql
   ALTER TABLE chunks ADD COLUMN IF NOT EXISTS img_emb vector(512);  -- CLIP embeddings
   ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_type TEXT DEFAULT 'text';  -- 'text' or 'image'
   ```

## Migration Path

1. **Phase 1** (Current): Text-only with OCR for images ✅
2. **Phase 2** (Recommended): Add CLIP embeddings for images
   - Install: `pip install sentence-transformers` (already installed)
   - Add CLIP model loading
   - Store image embeddings separately
   - Update retrieval to handle both text and image queries
3. **Phase 3** (Future): Unified multi-modal search
   - Combine text and image embeddings in retrieval
   - Support "find images like this" queries

## Installation for Multi-Modal

```bash
# Already installed
pip install sentence-transformers

# For CLIP-style embeddings
# (sentence-transformers includes CLIP models)
# No additional installation needed!

# Optional: For OpenCLIP
pip install open-clip-torch
```

## Summary

- **Current**: `BAAI/bge-m3` is perfect for text-only RAG ✅
- **For images**: Consider adding CLIP-style embeddings for true visual search
- **No API dependency**: All models run locally ✅
- **Easy upgrade**: Can add CLIP alongside existing text embeddings

