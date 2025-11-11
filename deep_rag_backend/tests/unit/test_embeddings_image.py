"""
Unit tests for image embedding functionality.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
from pathlib import Path
from ingestion.embeddings.image import embed_image


class TestEmbedImage:
    """Tests for embed_image function."""
    
    @patch('ingestion.embeddings.image.get_clip_model')
    @patch('ingestion.embeddings.image._validate_and_resize_image')
    def test_embed_image_with_path_string(self, mock_validate, mock_get_model):
        """Test embedding image from string path."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3] * 256)  # 768 dims
        mock_get_model.return_value = mock_model
        
        # Create a temporary image file path (mocked)
        with patch('ingestion.embeddings.image.Image.open') as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_validate.return_value = mock_image  # Return the same mock image after validation
            mock_open.return_value = mock_image
            
            result = embed_image("test_image.png", normalize_emb=False)
            
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert len(result) == 768
            mock_model.encode.assert_called_once()
            mock_open.assert_called_once_with("test_image.png")
            mock_validate.assert_called_once()
    
    @patch('ingestion.embeddings.image.get_clip_model')
    @patch('ingestion.embeddings.image._validate_and_resize_image')
    def test_embed_image_with_pathlib_path(self, mock_validate, mock_get_model):
        """Test embedding image from Path object."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3] * 256)
        mock_get_model.return_value = mock_model
        
        with patch('ingestion.embeddings.image.Image.open') as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_validate.return_value = mock_image
            mock_open.return_value = mock_image
            
            result = embed_image(Path("test_image.png"), normalize_emb=False)
            
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert len(result) == 768
            mock_model.encode.assert_called_once()
            mock_validate.assert_called_once()
    
    @patch('ingestion.embeddings.image.get_clip_model')
    @patch('ingestion.embeddings.image._validate_and_resize_image')
    def test_embed_image_with_pil_image(self, mock_validate, mock_get_model):
        """Test embedding image from PIL Image object."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3] * 256)
        mock_get_model.return_value = mock_model
        
        mock_pil_image = MagicMock(spec=Image.Image)
        mock_pil_image.convert.return_value = mock_pil_image
        mock_validate.return_value = mock_pil_image
        
        result = embed_image(mock_pil_image, normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 768
        mock_model.encode.assert_called_once()
        mock_pil_image.convert.assert_called_once_with('RGB')
        mock_validate.assert_called_once()
    
    @patch('ingestion.embeddings.image.get_clip_model')
    @patch('ingestion.embeddings.image.normalize')
    @patch('ingestion.embeddings.image._validate_and_resize_image')
    def test_embed_image_with_normalization(self, mock_validate, mock_normalize, mock_get_model):
        """Test that normalization is applied when normalize_emb=True."""
        mock_model = MagicMock()
        raw_embedding = np.array([0.1, 0.2, 0.3] * 256)
        mock_model.encode.return_value = raw_embedding
        mock_get_model.return_value = mock_model
        
        normalized_embedding = np.array([0.05, 0.1, 0.15] * 256)
        mock_normalize.return_value = normalized_embedding
        
        with patch('ingestion.embeddings.image.Image.open') as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_validate.return_value = mock_image
            mock_open.return_value = mock_image
            
            result = embed_image("test_image.png", normalize_emb=True)
            
            assert result is not None
            mock_normalize.assert_called_once_with(raw_embedding)
            np.testing.assert_array_equal(result, normalized_embedding)
            mock_validate.assert_called_once()
    
    @patch('ingestion.embeddings.image.get_clip_model')
    @patch('ingestion.embeddings.image._validate_and_resize_image')
    def test_embed_image_without_normalization(self, mock_validate, mock_get_model):
        """Test that normalization is skipped when normalize_emb=False."""
        mock_model = MagicMock()
        raw_embedding = np.array([0.1, 0.2, 0.3] * 256)
        mock_model.encode.return_value = raw_embedding
        mock_get_model.return_value = mock_model
        
        with patch('ingestion.embeddings.image.Image.open') as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_validate.return_value = mock_image
            mock_open.return_value = mock_image
            
            result = embed_image("test_image.png", normalize_emb=False)
            
            assert result is not None
            np.testing.assert_array_equal(result, raw_embedding)
            mock_validate.assert_called_once()
    
    def test_embed_image_with_unsupported_type(self):
        """Test that unsupported image types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported image type"):
            embed_image(123)  # Invalid type
    
    @patch('ingestion.embeddings.image.get_clip_model')
    @patch('ingestion.embeddings.image._validate_and_resize_image')
    def test_embed_image_encoding_failure(self, mock_validate, mock_get_model):
        """Test handling of encoding failures."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_get_model.return_value = mock_model
        
        with patch('ingestion.embeddings.image.Image.open') as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_validate.return_value = mock_image
            mock_open.return_value = mock_image
            
            with pytest.raises(Exception, match="Encoding failed"):
                embed_image("test_image.png")
            mock_validate.assert_called_once()

