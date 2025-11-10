"""
Semantic chunking utilities for PDF pages.
"""
import re
import os
import logging
import tempfile
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


def semantic_chunks(page_items: List[Dict], max_words: int = 25, overlap: int = 12) -> Tuple[List[Tuple], str]:
    """
    Split pages into chunks with multi-modal support.
    
    Note: CLIP-ViT-B/32 has max 77 tokens per text. 
    We use max_words=25 (words) to ensure chunks stay well under 77 tokens.
    Average word-to-token ratio is ~1.3-1.5, so 25 words ≈ 32-37 tokens (safe margin).
    
    Returns:
        Tuple of (chunks, temp_dir) where:
        - chunks: list of tuples (text, page_start, page_end, is_ocr, is_figure, content_type, image, image_path)
        - temp_dir: temporary directory path for saved images
        
    content_type: 'text', 'image', 'multimodal', 'pdf_text', 'pdf_image'
    image: PIL Image or None
    image_path: str path to saved image or None
    """
    chunks = []
    chunk_index = 0
    logger.info(f"Starting chunking with max_words={max_words} (CLIP-compatible, ~{int(max_words * 1.4)} tokens), overlap={overlap}")
    
    # Create temp directory for saving images
    temp_dir = tempfile.mkdtemp(prefix="pdf_images_")
    
    for p in page_items:
        page_num = p["page"]
        text = p["text"]
        images = p.get("images", [])
        captions = p.get("captions", [])
        is_ocr = p.get("is_ocr", False)
        
        # Process text chunks
        units = re.split(r'(?m)^(#+\s.*|[A-Z][^\n]{0,80}\n[-=]{3,}\s*$)|\n{2,}', text)
        units = [u for u in units if u and u.strip()]
        buf, count = [], 0
        
        for u in units:
            toks = len(u.split())  # Word count, not token count
            if count + toks > max_words and buf:
                chunk_text = " ".join(buf)
                content_type = 'pdf_text' if text else 'text'
                
                # Check if there are images on this page that might relate to this chunk
                # If images exist, we'll create multimodal chunks
                if images:
                    # For multimodal: combine text with first image
                    image = images[0] if images else None
                    image_path = None
                    if image:
                        # Save image temporarily
                        image_path = os.path.join(temp_dir, f"page_{page_num}_img_0.png")
                        image.save(image_path)
                        content_type = 'multimodal'
                    
                    chunks.append((chunk_text, page_num, page_num, is_ocr, False, content_type, image, image_path))
                else:
                    chunks.append((chunk_text, page_num, page_num, is_ocr, False, content_type, None, None))
                
                chunk_index += 1
                overlap_words = " ".join(chunk_text.split()[-overlap:])
                buf, count = [overlap_words, u], len(overlap_words.split()) + toks
            else:
                buf.append(u)
                count += toks
                
        # Final text chunk
        if buf:
            chunk_text = " ".join(buf)
            content_type = 'pdf_text' if text else 'text'
            
            if images:
                image = images[0] if images else None
                image_path = None
                if image:
                    image_path = os.path.join(temp_dir, f"page_{page_num}_img_final.png")
                    image.save(image_path)
                    content_type = 'multimodal'
                chunks.append((chunk_text, page_num, page_num, is_ocr, False, content_type, image, image_path))
            else:
                chunks.append((chunk_text, page_num, page_num, is_ocr, False, content_type, None, None))
            chunk_index += 1

        # Create separate chunks for images (if not already combined with text)
        for img_idx, image in enumerate(images):
            if len(images) > 1 or not text:  # Create separate image chunk if multiple images or no text
                image_path = os.path.join(temp_dir, f"page_{page_num}_img_{img_idx}.png")
                image.save(image_path)
                # Use caption if available, otherwise use image description
                caption_text = captions[img_idx] if img_idx < len(captions) else f"[Image {img_idx + 1} from page {page_num}]"
                chunks.append((caption_text, page_num, page_num, is_ocr, True, 'pdf_image', image, image_path))
                chunk_index += 1

        # Add caption chunks
        for cap in captions:
            chunks.append((cap, page_num, page_num, is_ocr, True, 'text', None, None))
            chunk_index += 1
    
    logger.info(f"Chunking complete: created {len(chunks)} chunks total")
    return chunks, temp_dir

