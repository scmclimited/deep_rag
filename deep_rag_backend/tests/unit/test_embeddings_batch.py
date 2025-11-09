"""
Unit tests for batch embedding functionality.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
from ingestion.embeddings.batch import embed_batch


class TestEmbedBatch:
    """Tests for embed_batch function."""
    
    @patch('ingestion.embeddings.batch.embed_text')
    def test_embed_batch_text_only(self, mock_embed_text):
        """Test embedding batch of text strings."""
        mock_embedding = np.array([0.1, 0.2, 0.3] * 256)  # 768 dims
        mock_embed_text.return_value = mock_embedding
        
        texts = ["text1", "text2", "text3"]
        result = embed_batch(texts, normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 768)
        assert mock_embed_text.call_count == 3
    
    @patch('ingestion.embeddings.batch.embed_image')
    def test_embed_batch_image_only(self, mock_embed_image):
        """Test embedding batch of PIL Images."""
        mock_embedding = np.array([0.1, 0.2, 0.3] * 256)
        mock_embed_image.return_value = mock_embedding
        
        mock_images = [MagicMock(spec=Image.Image), MagicMock(spec=Image.Image)]
        result = embed_batch(mock_images, normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 768)
        assert mock_embed_image.call_count == 2
    
    @patch('ingestion.embeddings.batch.embed_multi_modal')
    def test_embed_batch_multimodal_tuples(self, mock_embed_multi_modal):
        """Test embedding batch of (text, image) tuples."""
        mock_embedding = np.array([0.1, 0.2, 0.3] * 256)
        mock_embed_multi_modal.return_value = mock_embedding
        
        items = [
            ("text1", "image1.png"),
            ("text2", "image2.png")
        ]
        result = embed_batch(items, normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 768)
        assert mock_embed_multi_modal.call_count == 2
    
    @patch('ingestion.embeddings.batch.embed_text')
    @patch('ingestion.embeddings.batch.embed_image')
    @patch('ingestion.embeddings.batch.embed_multi_modal')
    def test_embed_batch_mixed_types(self, mock_embed_multi_modal, mock_embed_image, mock_embed_text):
        """Test embedding batch with mixed text, image, and multimodal items."""
        mock_embedding = np.array([0.1, 0.2, 0.3] * 256)
        mock_embed_text.return_value = mock_embedding
        mock_embed_image.return_value = mock_embedding
        mock_embed_multi_modal.return_value = mock_embedding
        
        items = [
            "text1",
            MagicMock(spec=Image.Image),
            ("text2", "image.png")
        ]
        result = embed_batch(items, normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 768)
        mock_embed_text.assert_called_once()
        mock_embed_image.assert_called_once()
        mock_embed_multi_modal.assert_called_once()
    
    @patch('ingestion.embeddings.batch.embed_text')
    @patch('ingestion.embeddings.batch.normalize')
    def test_embed_batch_with_normalization(self, mock_normalize, mock_embed_text):
        """Test that normalization is applied when normalize_emb=True."""
        raw_embedding = np.array([0.1, 0.2, 0.3] * 256)
        normalized_embedding = np.array([0.05, 0.1, 0.15] * 256)
        
        mock_embed_text.return_value = raw_embedding
        mock_normalize.return_value = normalized_embedding
        
        texts = ["text1", "text2"]
        result = embed_batch(texts, normalize_emb=True)
        
        assert result is not None
        assert result.shape == (2, 768)
        assert mock_normalize.call_count == 2
    
    @patch('ingestion.embeddings.batch.embed_text')
    def test_embed_batch_without_normalization(self, mock_embed_text):
        """Test that normalization is skipped when normalize_emb=False."""
        raw_embedding = np.array([0.1, 0.2, 0.3] * 256)
        mock_embed_text.return_value = raw_embedding
        
        texts = ["text1", "text2"]
        result = embed_batch(texts, normalize_emb=False)
        
        assert result is not None
        assert result.shape == (2, 768)
        # Verify raw embeddings are used (no normalization)
        np.testing.assert_array_equal(result[0], raw_embedding)
    
    def test_embed_batch_unsupported_type(self):
        """Test that unsupported item types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported item type"):
            embed_batch([123])  # Invalid type
    
    @patch('ingestion.embeddings.batch.embed_text')
    def test_embed_batch_empty_list(self, mock_embed_text):
        """Test embedding empty list."""
        result = embed_batch([], normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        # Empty array has shape (0,) not (0, 768) - this is expected behavior
        assert result.shape == (0,)
        mock_embed_text.assert_not_called()

