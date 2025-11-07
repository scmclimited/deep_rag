"""
Unit tests for multi-modal embedding functionality.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
from pathlib import Path
from ingestion.embeddings.multimodal import embed_multi_modal


class TestEmbedMultiModal:
    """Tests for embed_multi_modal function."""
    
    @patch('ingestion.embeddings.multimodal.embed_text')
    def test_embed_multi_modal_text_only(self, mock_embed_text):
        """Test embedding text only."""
        mock_embedding = np.array([0.1, 0.2, 0.3] * 256)  # 768 dims
        mock_embed_text.return_value = mock_embedding
        
        result = embed_multi_modal(text="test text", normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 768
        # embed_text is called with positional args: embed_text(text, normalize_emb)
        mock_embed_text.assert_called_once()
        call_args = mock_embed_text.call_args
        assert call_args[0][0] == "test text"
        assert call_args[0][1] == False  # normalize_emb=False
    
    @patch('ingestion.embeddings.multimodal.embed_image')
    def test_embed_multi_modal_image_only(self, mock_embed_image):
        """Test embedding image only."""
        mock_embedding = np.array([0.1, 0.2, 0.3] * 256)
        mock_embed_image.return_value = mock_embedding
        
        result = embed_multi_modal(image_path="test_image.png", normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 768
        # embed_image is called with positional args: embed_image(image_path, normalize_emb)
        mock_embed_image.assert_called_once()
        call_args = mock_embed_image.call_args
        assert call_args[0][0] == "test_image.png"
        assert call_args[0][1] == False  # normalize_emb=False
    
    @patch('ingestion.embeddings.multimodal.embed_text')
    @patch('ingestion.embeddings.multimodal.get_clip_model')
    @patch('ingestion.embeddings.multimodal.normalize')
    def test_embed_multi_modal_text_and_image(self, mock_normalize, mock_get_model, mock_embed_text):
        """Test embedding text and image together."""
        text_emb = np.array([0.1, 0.2, 0.3] * 256)
        image_emb = np.array([0.4, 0.5, 0.6] * 256)
        combined_emb = (text_emb + image_emb) / 2.0
        normalized_emb = np.array([0.25, 0.35, 0.45] * 256)
        
        mock_embed_text.return_value = text_emb
        mock_model = MagicMock()
        mock_model.encode.return_value = image_emb
        mock_get_model.return_value = mock_model
        mock_normalize.return_value = normalized_emb
        
        with patch('ingestion.embeddings.multimodal.Image.open') as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image
            
            result = embed_multi_modal(text="test text", image_path="test_image.png", normalize_emb=True)
            
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert len(result) == 768
            mock_embed_text.assert_called_once_with("test text", normalize_emb=False, max_length=77)
            mock_model.encode.assert_called_once()
            mock_normalize.assert_called_once()
            np.testing.assert_array_equal(result, normalized_emb)
    
    @patch('ingestion.embeddings.multimodal.embed_text')
    @patch('ingestion.embeddings.multimodal.get_clip_model')
    def test_embed_multi_modal_text_and_image_no_normalize(self, mock_get_model, mock_embed_text):
        """Test embedding text and image without normalization."""
        text_emb = np.array([0.1, 0.2, 0.3] * 256)
        image_emb = np.array([0.4, 0.5, 0.6] * 256)
        combined_emb = (text_emb + image_emb) / 2.0
        
        mock_embed_text.return_value = text_emb
        mock_model = MagicMock()
        mock_model.encode.return_value = image_emb
        mock_get_model.return_value = mock_model
        
        with patch('ingestion.embeddings.multimodal.Image.open') as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image
            
            result = embed_multi_modal(text="test text", image_path="test_image.png", normalize_emb=False)
            
            assert result is not None
            np.testing.assert_array_almost_equal(result, combined_emb)
    
    @patch('ingestion.embeddings.multimodal.embed_text')
    @patch('ingestion.embeddings.multimodal.get_clip_model')
    def test_embed_multi_modal_with_pil_image(self, mock_get_model, mock_embed_text):
        """Test embedding with PIL Image object."""
        text_emb = np.array([0.1, 0.2, 0.3] * 256)
        image_emb = np.array([0.4, 0.5, 0.6] * 256)
        
        mock_embed_text.return_value = text_emb
        mock_model = MagicMock()
        mock_model.encode.return_value = image_emb
        mock_get_model.return_value = mock_model
        
        mock_pil_image = MagicMock(spec=Image.Image)
        mock_pil_image.convert.return_value = mock_pil_image
        
        result = embed_multi_modal(text="test text", image_path=mock_pil_image, normalize_emb=False)
        
        assert result is not None
        mock_pil_image.convert.assert_called_once_with('RGB')
        mock_model.encode.assert_called_once()
    
    def test_embed_multi_modal_no_inputs(self):
        """Test that ValueError is raised when neither text nor image is provided."""
        with pytest.raises(ValueError, match="Must provide either text or image_path"):
            embed_multi_modal()
    
    @patch('ingestion.embeddings.multimodal.embed_text')
    @patch('ingestion.embeddings.multimodal.get_clip_model')
    def test_embed_multi_modal_unsupported_image_type(self, mock_get_model, mock_embed_text):
        """Test that unsupported image types raise ValueError."""
        text_emb = np.array([0.1, 0.2, 0.3] * 256)
        mock_embed_text.return_value = text_emb
        
        with pytest.raises(ValueError, match="Unsupported image type"):
            embed_multi_modal(text="test text", image_path=123)

