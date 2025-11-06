# ingest_unified.py
# Unified ingestion entry point that handles all file types:
# - PDF (text, images, or mixed)
# - Plain text files (.txt)
# - Image files (.png, .jpg, .jpeg)
#
# Automatically detects file type and routes to appropriate handler

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Import specialized handlers
from ingestion.ingest import ingest as ingest_pdf
from ingestion.ingest_text import ingest_text_file
from ingestion.ingest_image import ingest_image

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    'pdf': ['.pdf'],
    'text': ['.txt', '.text', '.md', '.markdown'],
    'image': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
}

def get_file_type(file_path: str) -> Optional[str]:
    """
    Determine file type based on extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        'pdf', 'text', 'image', or None if unsupported
    """
    ext = Path(file_path).suffix.lower()
    
    for file_type, extensions in SUPPORTED_EXTENSIONS.items():
        if ext in extensions:
            return file_type
    
    return None

def ingest_file(file_path: str, title: Optional[str] = None):
    """
    Unified ingestion function that handles all supported file types.
    
    Automatically detects file type and routes to appropriate handler:
    - PDF → ingest.py (handles text + images + OCR)
    - Text → ingest_text.py
    - Image → ingest_image.py (handles OCR + multi-modal embedding)
    
    Args:
        file_path: Path to file (PDF, TXT, PNG, JPG, JPEG, etc.)
        title: Optional document title
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is not supported
    """
    # Verify file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Detect file type
    file_type = get_file_type(file_path)
    
    if file_type is None:
        supported = []
        for exts in SUPPORTED_EXTENSIONS.values():
            supported.extend(exts)
        raise ValueError(
            f"Unsupported file type: {Path(file_path).suffix}\n"
            f"Supported extensions: {', '.join(supported)}"
        )
    
    # Log ingestion attempt
    logger.info(f"Ingesting file: {file_path}")
    logger.info(f"Detected file type: {file_type}")
    logger.info(f"Title: {title or 'Auto-detect'}")
    
    # Route to appropriate handler
    try:
        if file_type == 'pdf':
            logger.info("Routing to PDF handler (supports text, images, OCR)")
            ingest_pdf(file_path, title=title)
            
        elif file_type == 'text':
            logger.info("Routing to text file handler")
            ingest_text_file(file_path, title=title)
            
        elif file_type == 'image':
            logger.info("Routing to image handler (supports OCR + multi-modal embedding)")
            ingest_image(file_path, title=title)
            
        else:
            raise ValueError(f"Handler not found for file type: {file_type}")
        
        logger.info(f"Successfully ingested: {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to ingest {file_path}: {e}", exc_info=True)
        raise

def is_supported(file_path: str) -> bool:
    """
    Check if a file is supported for ingestion.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file type is supported, False otherwise
    """
    return get_file_type(file_path) is not None

def list_supported_extensions() -> dict:
    """
    Get dictionary of supported file extensions by category.
    
    Returns:
        Dict mapping file type to list of extensions
    """
    return SUPPORTED_EXTENSIONS.copy()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ingest_unified.py <file_path> [title]")
        print("\nSupported file types:")
        for file_type, extensions in SUPPORTED_EXTENSIONS.items():
            print(f"  {file_type.upper()}: {', '.join(extensions)}")
        sys.exit(1)
    
    file_path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        ingest_file(file_path, title=title)
        print(f"\n✓ Successfully ingested: {file_path}")
    except Exception as e:
        print(f"\n✗ Failed to ingest {file_path}: {e}")
        sys.exit(1)

