"""
Vector parsing utilities for pgvector.
"""
import re
import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)


def parse_vector(emb: Union[str, list, tuple, np.ndarray]) -> np.ndarray:
    """
    Parse pgvector vector type from database to numpy array.
    
    Handles multiple input formats:
    - String format: '[0.1,0.2,0.3]'
    - List/tuple format: [0.1, 0.2, 0.3]
    - Numpy array: np.array([0.1, 0.2, 0.3])
    
    Also handles scientific notation issues (e.g., "3.088634-05" -> "3.088634e-05")
    
    Args:
        emb: Vector in any supported format
        
    Returns:
        Numpy array of float32 values
        
    Raises:
        ValueError: If vector cannot be parsed
    """
    if isinstance(emb, str):
        # pgvector returns vectors as strings like '[0.1,0.2,0.3]'
        # Remove brackets and split by comma
        try:
            # Remove brackets and whitespace
            emb_str = emb.strip('[]').strip()
            # Split by comma and convert to float
            # Handle scientific notation issues (e.g., "3.088634-05" -> "3.088634e-05")
            parts = [p.strip() for p in emb_str.split(',')]
            values = []
            for part in parts:
                # Fix malformed scientific notation (missing 'e' before exponent)
                # Pattern: number followed by - or + followed by digits (exponent)
                # Match patterns like "3.088634-05" and convert to "3.088634e-05"
                part = re.sub(r'([0-9])([+-])([0-9]+)$', r'\1e\2\3', part)
                values.append(float(part))
            return np.array(values, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to parse vector: {emb[:100]}... Error: {e}")
            raise ValueError(f"Could not parse vector from database: {e}")
    elif isinstance(emb, (list, tuple)):
        # Already a list/tuple, convert to numpy array
        return np.array(emb, dtype=np.float32)
    elif isinstance(emb, np.ndarray):
        # Already a numpy array
        return emb.astype(np.float32)
    else:
        raise ValueError(f"Unexpected vector type: {type(emb)}")

