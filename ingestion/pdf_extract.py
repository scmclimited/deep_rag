"""
PDF extraction utilities.
"""
import re
import logging
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import io
from typing import List, Dict

logger = logging.getLogger(__name__)


def pdf_extract(path: str, extract_images: bool = True) -> List[Dict]:
    """
    Extract pagewise text, OCR fallback if blank; extract images if present.
    
    Returns:
        List of page dicts with:
        - page: page number
        - text: extracted text
        - images: list of PIL Images found on page
        - captions: figure captions
        - is_ocr: whether OCR was used
    """
    doc = fitz.open(path)
    pages = []
    
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        text = re.sub(r'[ \t]+', ' ', text).strip()
        is_scan = (len(text) < 20)
        ocr_text = ""
        if is_scan:
            # OCR at page-level
            images = convert_from_path(path, first_page=i+1, last_page=i+1, dpi=300)
            ocr_text = pytesseract.image_to_string(images[0])
        final_text = text if len(text) >= len(ocr_text) else ocr_text

        # Extract images from PDF page
        page_images = []
        if extract_images:
            try:
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                        page_images.append(image)
                        logger.debug(f"Extracted image {img_index} from page {i+1}")
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {i+1}: {e}")
            except Exception as e:
                logger.warning(f"Failed to extract images from page {i+1}: {e}")

        # Extract figure captions
        captions = []
        for block in page.get_text("blocks"):
            btxt = block[4].strip()
            if re.search(r'^(Figure|Fig\.|Diagram)\s*\d+', btxt, re.I):
                captions.append(btxt)

        pages.append({
            "page": i+1,
            "text": final_text,
            "images": page_images,
            "captions": captions,
            "is_ocr": is_scan and len(final_text) > 0
        })
    
    doc.close()
    return pages

