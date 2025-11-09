"""
Title extraction utilities for PDFs.
"""
import logging
from pathlib import Path
import fitz  # PyMuPDF
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def extract_title(pdf_path: str, pages: List[Dict], provided_title: Optional[str] = None) -> str:
    """
    Extract title from PDF metadata, first page, or use filename.
    
    Args:
        pdf_path: Path to PDF file
        pages: List of page dicts from pdf_extract
        provided_title: Optional provided title
        
    Returns:
        Extracted or determined title
    """
    # Determine title: use provided title, or extract from PDF, or use filename
    if provided_title:
        # Truncate provided title to 50 words max to avoid token issues
        words = provided_title.split()
        if len(words) > 20:
            final_title = " ".join(words[:20])
            print(f"Provided title truncated from {len(words)} words to 20 words")
        else:
            final_title = provided_title
        print(f"Using provided title: {final_title}")
        return final_title
    
    # Try to extract title from PDF metadata or first page
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        if metadata.get("title"):
            # Truncate metadata title to 20 words max
            metadata_title = metadata["title"]
            words = metadata_title.split()
            if len(words) > 20:
                final_title = " ".join(words[:50])
                print(f"Metadata title truncated from {len(words)} words to 50 words")
            else:
                final_title = metadata_title
            print(f"Extracted title from PDF metadata: {final_title}")
            doc.close()
            return final_title
        elif pages and pages[0].get("text"):
            # Extract first line or first 100 chars as title
            # Truncate to 50 words max to avoid CLIP token issues if title is embedded elsewhere
            first_text = pages[0]["text"].strip()
            first_line = first_text.split("\n")[0][:100].strip()
            # Further truncate to 50 words if it's very long
            words = first_line.split()
            if len(words) > 20:
                first_line = " ".join(words[:20])
                print(f"Title truncated from {len(words)} words to 20 words")
            final_title = first_line if first_line else Path(pdf_path).stem
            if first_line:
                print(f"Extracted title from first page: {final_title}")
            else:
                print(f"Using filename as title: {final_title}")
            doc.close()
            return final_title
        else:
            final_title = Path(pdf_path).stem  # filename without extension
            print(f"Using filename as title: {final_title}")
            doc.close()
            return final_title
    except Exception as e:
        final_title = Path(pdf_path).stem
        print(f"Could not extract title from PDF, using filename: {final_title} (error: {e})")
        return final_title

