"""
Unit tests for retrieval.vector_utils module.
"""
import pytest
import numpy as np
from retrieval.vector_utils import parse_vector


def test_parse_vector_from_string():
    """Test parsing vector from string format."""
    vector_str = "[0.1, 0.2, 0.3]"
    result = parse_vector(vector_str)
    
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == 3
    assert np.allclose(result, [0.1, 0.2, 0.3])


def test_parse_vector_from_list():
    """Test parsing vector from list."""
    vector_list = [0.1, 0.2, 0.3]
    result = parse_vector(vector_list)
    
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == 3
    assert np.allclose(result, [0.1, 0.2, 0.3])


def test_parse_vector_from_numpy():
    """Test parsing vector from numpy array."""
    vector_np = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    result = parse_vector(vector_np)
    
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == 3
    assert np.allclose(result, [0.1, 0.2, 0.3])


def test_parse_vector_scientific_notation():
    """Test parsing vector with scientific notation fix."""
    # Test malformed scientific notation (missing 'e')
    vector_str = "[1.0, 2.0, 3.088634-05]"
    result = parse_vector(vector_str)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    # Should handle the scientific notation fix
    assert result[2] < 1.0  # Should be a very small number

