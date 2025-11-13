#!/usr/bin/env python3
"""
Download CLIP and reranker models from Hugging Face to local cache.
This allows us to use the models without downloading them every Docker rebuild.

Prerequisites:
    pip install transformers sentence-transformers torch
    
    Or install minimal requirements:
    pip install -r requirements-download.txt

Usage:
    python scripts/download_model.py                    # Download both CLIP and reranker
    python scripts/download_model.py --clip-only       # Download only CLIP model
    python scripts/download_model.py --reranker-only  # Download only reranker model
    python scripts/download_model.py --model openai/clip-vit-large-patch14-336 --cache-dir ./models/clip
"""
import os
import sys
from pathlib import Path

# Check if required modules are installed
try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    print("❌ Error: 'transformers' module not found.", file=sys.stderr)
    print("\nPlease install dependencies first:", file=sys.stderr)
    print("  pip install transformers sentence-transformers torch", file=sys.stderr)
    print("\nOr use minimal requirements:", file=sys.stderr)
    print("  pip install -r requirements-download.txt", file=sys.stderr)
    sys.exit(1)

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print("❌ Error: 'sentence-transformers' module not found.", file=sys.stderr)
    print("\nPlease install dependencies first:", file=sys.stderr)
    print("  pip install sentence-transformers", file=sys.stderr)
    sys.exit(1)

def download_model(model_name: str, cache_dir: str = None):
    """
    Download CLIP model and processor to local directory.
    
    Args:
        model_name: Hugging Face model identifier (e.g., 'openai/clip-vit-large-patch14-336')
        cache_dir: Local directory to store the model (default: ./models/{model_name})
    
    Returns:
        Path to the downloaded model directory
    """
    if cache_dir is None:
        # Use a models directory in the backend
        cache_dir = f"./models/{model_name.replace('/', '_')}"
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model '{model_name}' to '{cache_dir}'...")
    print(f"This may take a few minutes (model size: ~3.4 GB)...")
    
    try:
        # Download and save model and processor directly to the cache directory
        print("Downloading model...")
        model = CLIPModel.from_pretrained(model_name)
        print("Downloading processor...")
        processor = CLIPProcessor.from_pretrained(model_name)
        
        # Save to the cache directory explicitly
        print(f"Saving model to {cache_path}...")
        model.save_pretrained(str(cache_path))
        processor.save_pretrained(str(cache_path))
        
        print(f"✅ Model downloaded successfully to: {cache_path.absolute()}")
        print(f"   Model files are in: {cache_path}")
        print(f"\nTo use this model in Docker, set in your .env file:")
        print(f"   CLIP_MODEL_PATH={cache_path.absolute()}")
        return str(cache_path.absolute())
    except Exception as e:
        print(f"❌ Error downloading model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

def download_reranker_model(model_name: str = None, cache_dir: str = None):
    """
    Download reranker model (CrossEncoder) to local directory.
    
    Args:
        model_name: Hugging Face model identifier (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        cache_dir: Local directory to store the model (default: ./models/{model_name})
    
    Returns:
        Path to the downloaded model directory
    """
    if model_name is None:
        model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    if cache_dir is None:
        cache_dir = f"./models/{model_name.replace('/', '_')}"
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading reranker model '{model_name}' to '{cache_dir}'...")
    print(f"This may take a few minutes (model size: ~100-200 MB)...")
    
    try:
        # Download and save reranker model
        print("Downloading reranker...")
        reranker = CrossEncoder(model_name)
        
        # Save to the cache directory explicitly
        print(f"Saving reranker to {cache_path}...")
        reranker.save(str(cache_path))
        
        print(f"✅ Reranker model downloaded successfully to: {cache_path.absolute()}")
        print(f"   Model files are in: {cache_path}")
        print(f"\nTo use this model in Docker, set in your .env file:")
        print(f"   RERANK_MODEL_PATH={cache_path.absolute()}")
        return str(cache_path.absolute())
    except Exception as e:
        print(f"❌ Error downloading reranker model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download CLIP and reranker models from Hugging Face")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("CLIP_MODEL", "openai/clip-vit-large-patch14-336"),
        help="CLIP model name to download (default: from CLIP_MODEL env var or openai/clip-vit-large-patch14-336)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to store the CLIP model (default: ./models/{model_name})"
    )
    parser.add_argument(
        "--clip-only",
        action="store_true",
        help="Download only the CLIP model"
    )
    parser.add_argument(
        "--reranker-only",
        action="store_true",
        help="Download only the reranker model"
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="Reranker model name (default: from RERANK_MODEL env var or cross-encoder/ms-marco-MiniLM-L-6-v2)"
    )
    
    args = parser.parse_args()
    
    if args.reranker_only:
        download_reranker_model(args.reranker_model)
    elif args.clip_only:
        download_model(args.model, args.cache_dir)
    else:
        # Download both models
        print("=" * 60)
        print("Downloading CLIP model...")
        print("=" * 60)
        download_model(args.model, args.cache_dir)
        
        print("\n" + "=" * 60)
        print("Downloading reranker model...")
        print("=" * 60)
        download_reranker_model(args.reranker_model)
        
        print("\n" + "=" * 60)
        print("✅ All models downloaded successfully!")
        print("=" * 60)

