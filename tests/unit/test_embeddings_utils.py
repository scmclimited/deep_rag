"""
Unit tests for embedding utility functions.
"""
import pytest
import numpy as np
from ingestion.embeddings.utils import normalize


class TestNormalize:
    """Tests for normalize function."""
    
    def test_normalize_basic(self):
        """Test basic normalization."""
        v = np.array([3.0, 4.0, 0.0])
        result = normalize(v)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        # Norm of [3, 4, 0] is 5, so normalized should be [0.6, 0.8, 0.0]
        expected = np.array([0.6, 0.8, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_normalize_unit_vector(self):
        """Test normalizing a unit vector (should remain unchanged)."""
        v = np.array([1.0, 0.0, 0.0])
        result = normalize(v)
        
        np.testing.assert_array_almost_equal(result, v)
    
    def test_normalize_zero_vector(self):
        """Test normalizing a zero vector (should use epsilon to avoid division by zero)."""
        v = np.array([0.0, 0.0, 0.0])
        result = normalize(v)
        
        assert result is not None
        # Should divide by 1e-12 (epsilon) to avoid division by zero
        expected = v / 1e-12
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_normalize_negative_values(self):
        """Test normalizing vector with negative values."""
        v = np.array([-3.0, -4.0, 0.0])
        result = normalize(v)
        
        # Norm is still 5, normalized should be [-0.6, -0.8, 0.0]
        expected = np.array([-0.6, -0.8, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_normalize_embedding_dimension(self):
        """Test normalizing a 768-dimensional embedding vector."""
        v = np.random.rand(768)
        result = normalize(v)
        
        assert result is not None
        assert len(result) == 768
        # Check that result has unit norm (approximately)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6
    
    def test_normalize_small_vector(self):
        """Test normalizing a very small vector."""
        v = np.array([1e-10, 1e-10, 1e-10])
        result = normalize(v)
        
        assert result is not None
        # Should still normalize correctly
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6
    
    def test_normalize_large_vector(self):
        """Test normalizing a large vector."""
        v = np.array([1000.0, 2000.0, 3000.0])
        result = normalize(v)
        
        assert result is not None
        # Should normalize to unit vector
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6
    
    def test_normalize_preserves_direction(self):
        """Test that normalization preserves direction."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = v1 * 5.0  # Same direction, different magnitude
        
        result1 = normalize(v1)
        result2 = normalize(v2)
        
        # Should be the same (within floating point precision)
        np.testing.assert_array_almost_equal(result1, result2)

