"""
Unit tests for image embedding functionality.
"""
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
from pathlib import Path
from ingestion.embeddings.image import embed_image


class TestEmbedImage:
    """Tests for embed_image function."""
    
    @patch('ingestion.embeddings.image.get_clip_processor')
    @patch('ingestion.embeddings.image.get_clip_model')
    @patch('ingestion.embeddings.image._validate_and_resize_image')
    @patch('ingestion.embeddings.image.torch.no_grad')
    def test_embed_image_with_path_string(self, mock_no_grad, mock_validate, mock_get_model, mock_get_processor):
        """Test embedding image from string path."""
        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        # Mock image features (torch tensor that converts to numpy)
        # Create a mock tensor that supports [0].cpu().numpy() chain
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([0.1, 0.2, 0.3] * 256)  # 768 dims
        mock_image_features = MagicMock()
        mock_image_features.__getitem__ = MagicMock(return_value=mock_tensor)
        mock_model.get_image_features.return_value = mock_image_features
        
        # Mock processor to return a dict
        mock_processor.return_value = {"pixel_values": torch.tensor([[1, 2, 3]])}
        
        mock_get_model.return_value = mock_model
        mock_get_processor.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        # Create a temporary image file path (mocked)
        with patch('ingestion.embeddings.image.Image.open') as mock_open:
            mock_image = Image.new('RGB', (100, 100))  # Real PIL Image
            mock_validate.return_value = mock_image
            mock_open.return_value = mock_image
            
            result = embed_image("test_image.png", normalize_emb=False)
            
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert len(result) == 768
            mock_processor.assert_called_once()
            mock_model.get_image_features.assert_called_once()
            mock_open.assert_called_once_with("test_image.png")
            mock_validate.assert_called_once()
    
    @patch('ingestion.embeddings.image.get_clip_processor')
    @patch('ingestion.embeddings.image.get_clip_model')
    @patch('ingestion.embeddings.image._validate_and_resize_image')
    @patch('ingestion.embeddings.image.torch.no_grad')
    def test_embed_image_with_pathlib_path(self, mock_no_grad, mock_validate, mock_get_model, mock_get_processor):
        """Test embedding image from Path object."""
        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        # Mock image features
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([0.1, 0.2, 0.3] * 256)
        mock_image_features = MagicMock()
        mock_image_features.__getitem__ = MagicMock(return_value=mock_tensor)
        mock_model.get_image_features.return_value = mock_image_features
        mock_processor.return_value = {"pixel_values": torch.tensor([[1, 2, 3]])}
        
        mock_get_model.return_value = mock_model
        mock_get_processor.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch('ingestion.embeddings.image.Image.open') as mock_open:
            mock_image = Image.new('RGB', (100, 100))
            mock_validate.return_value = mock_image
            mock_open.return_value = mock_image
            
            result = embed_image(Path("test_image.png"), normalize_emb=False)
            
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert len(result) == 768
            mock_model.get_image_features.assert_called_once()
            mock_validate.assert_called_once()
    
    @patch('ingestion.embeddings.image.get_clip_processor')
    @patch('ingestion.embeddings.image.get_clip_model')
    @patch('ingestion.embeddings.image._validate_and_resize_image')
    @patch('ingestion.embeddings.image.torch.no_grad')
    def test_embed_image_with_pil_image(self, mock_no_grad, mock_validate, mock_get_model, mock_get_processor):
        """Test embedding image from PIL Image object."""
        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        # Mock image features
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([0.1, 0.2, 0.3] * 256)
        mock_image_features = MagicMock()
        mock_image_features.__getitem__ = MagicMock(return_value=mock_tensor)
        mock_model.get_image_features.return_value = mock_image_features
        mock_processor.return_value = {"pixel_values": torch.tensor([[1, 2, 3]])}
        
        mock_get_model.return_value = mock_model
        mock_get_processor.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        mock_pil_image = Image.new('RGB', (100, 100))  # Real PIL Image
        mock_validate.return_value = mock_pil_image
        
        result = embed_image(mock_pil_image, normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 768
        mock_model.get_image_features.assert_called_once()
        mock_validate.assert_called_once()
    
    @patch('ingestion.embeddings.image.get_clip_processor')
    @patch('ingestion.embeddings.image.get_clip_model')
    @patch('ingestion.embeddings.image.normalize')
    @patch('ingestion.embeddings.image._validate_and_resize_image')
    @patch('ingestion.embeddings.image.torch.no_grad')
    def test_embed_image_with_normalization(self, mock_no_grad, mock_validate, mock_normalize, mock_get_model, mock_get_processor):
        """Test that normalization is applied when normalize_emb=True."""
        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        raw_embedding = np.array([0.1, 0.2, 0.3] * 256)
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = raw_embedding
        mock_image_features = MagicMock()
        mock_image_features.__getitem__ = MagicMock(return_value=mock_tensor)
        mock_model.get_image_features.return_value = mock_image_features
        mock_processor.return_value = {"pixel_values": torch.tensor([[1, 2, 3]])}
        
        mock_get_model.return_value = mock_model
        mock_get_processor.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        normalized_embedding = np.array([0.05, 0.1, 0.15] * 256)
        mock_normalize.return_value = normalized_embedding
        
        with patch('ingestion.embeddings.image.Image.open') as mock_open:
            mock_image = Image.new('RGB', (100, 100))
            mock_validate.return_value = mock_image
            mock_open.return_value = mock_image
            
            result = embed_image("test_image.png", normalize_emb=True)
            
            assert result is not None
            mock_normalize.assert_called_once()
            np.testing.assert_array_equal(result, normalized_embedding)
            mock_validate.assert_called_once()
    
    @patch('ingestion.embeddings.image.get_clip_processor')
    @patch('ingestion.embeddings.image.get_clip_model')
    @patch('ingestion.embeddings.image._validate_and_resize_image')
    @patch('ingestion.embeddings.image.torch.no_grad')
    def test_embed_image_without_normalization(self, mock_no_grad, mock_validate, mock_get_model, mock_get_processor):
        """Test that normalization is skipped when normalize_emb=False."""
        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        raw_embedding = np.array([0.1, 0.2, 0.3] * 256)
        mock_image_features = MagicMock()
        mock_image_features.__getitem__.return_value.cpu.return_value.numpy.return_value = raw_embedding
        mock_model.get_image_features.return_value = mock_image_features
        mock_processor.return_value = {"pixel_values": torch.tensor([[1, 2, 3]])}
        
        mock_get_model.return_value = mock_model
        mock_get_processor.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch('ingestion.embeddings.image.Image.open') as mock_open:
            mock_image = Image.new('RGB', (100, 100))
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
    
    @patch('ingestion.embeddings.image.get_clip_processor')
    @patch('ingestion.embeddings.image.get_clip_model')
    @patch('ingestion.embeddings.image._validate_and_resize_image')
    @patch('ingestion.embeddings.image.torch.no_grad')
    def test_embed_image_encoding_failure(self, mock_no_grad, mock_validate, mock_get_model, mock_get_processor):
        """Test handling of encoding failures."""
        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.side_effect = Exception("Encoding failed")
        
        mock_get_model.return_value = mock_model
        mock_get_processor.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch('ingestion.embeddings.image.Image.open') as mock_open:
            mock_image = Image.new('RGB', (100, 100))
            mock_validate.return_value = mock_image
            mock_open.return_value = mock_image
            
            with pytest.raises(ValueError, match="Failed to encode image"):
                embed_image("test_image.png")
            mock_validate.assert_called_once()

