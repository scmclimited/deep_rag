#!/usr/bin/env python3
"""
Clean Python cache files (__pycache__ directories and .pyc files).
Can be run via: python -m scripts.clean_cache or via make clean-cache
"""
import os
import sys
from pathlib import Path


def clean_cache(root_dir: str = None) -> int:
    """
    Remove all __pycache__ directories and .pyc/.pyo files.
    
    Args:
        root_dir: Root directory to clean (defaults to script's parent directory)
        
    Returns:
        Number of items removed
    """
    if root_dir is None:
        # Default to backend directory (parent of scripts/)
        root_dir = Path(__file__).parent.parent
    
    root_path = Path(root_dir).resolve()
    if not root_path.exists():
        print(f"Error: Directory {root_path} does not exist")
        return 0
    
    removed_count = 0
    
    # Remove __pycache__ directories
    for pycache_dir in root_path.rglob("__pycache__"):
        try:
            import shutil
            shutil.rmtree(pycache_dir)
            removed_count += 1
            print(f"Removed: {pycache_dir}")
        except Exception as e:
            print(f"Error removing {pycache_dir}: {e}")
    
    # Remove .pyc files
    for pyc_file in root_path.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            removed_count += 1
            print(f"Removed: {pyc_file}")
        except Exception as e:
            print(f"Error removing {pyc_file}: {e}")
    
    # Remove .pyo files
    for pyo_file in root_path.rglob("*.pyo"):
        try:
            pyo_file.unlink()
            removed_count += 1
            print(f"Removed: {pyo_file}")
        except Exception as e:
            print(f"Error removing {pyo_file}: {e}")
    
    return removed_count


def main():
    """Entry point for script."""
    root = sys.argv[1] if len(sys.argv) > 1 else None
    count = clean_cache(root)
    if count > 0:
        print(f"\n✓ Cleaned {count} cache item(s)")
    else:
        print("\n✓ No cache files found")
    return 0 if count >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())

