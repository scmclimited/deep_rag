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
    
    @patch('ingestion.embeddings.multimodal.embed_image')
    @patch('ingestion.embeddings.multimodal.embed_text')
    @patch('ingestion.embeddings.utils.normalize')
    def test_embed_multi_modal_text_and_image(self, mock_normalize, mock_embed_text, mock_embed_image):
        """Test embedding text and image together."""
        text_emb = np.array([0.1, 0.2, 0.3] * 256)
        image_emb = np.array([0.4, 0.5, 0.6] * 256)
        combined_emb = (text_emb + image_emb) / 2.0
        normalized_emb = np.array([0.25, 0.35, 0.45] * 256)
        
        mock_embed_text.return_value = text_emb
        mock_embed_image.return_value = image_emb
        mock_normalize.return_value = normalized_emb
        
        result = embed_multi_modal(text="test text", image_path="test_image.png", normalize_emb=True)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 768
        mock_embed_text.assert_called_once_with("test text", normalize_emb=False, max_length=77)
        mock_embed_image.assert_called_once_with("test_image.png", normalize_emb=False)
        mock_normalize.assert_called_once()
        np.testing.assert_array_equal(result, normalized_emb)
    
    @patch('ingestion.embeddings.multimodal.embed_image')
    @patch('ingestion.embeddings.multimodal.embed_text')
    def test_embed_multi_modal_text_and_image_no_normalize(self, mock_embed_text, mock_embed_image):
        """Test embedding text and image without normalization."""
        text_emb = np.array([0.1, 0.2, 0.3] * 256)
        image_emb = np.array([0.4, 0.5, 0.6] * 256)
        combined_emb = (text_emb + image_emb) / 2.0
        
        mock_embed_text.return_value = text_emb
        mock_embed_image.return_value = image_emb
        
        result = embed_multi_modal(text="test text", image_path="test_image.png", normalize_emb=False)
        
        assert result is not None
        np.testing.assert_array_almost_equal(result, combined_emb)
    
    @patch('ingestion.embeddings.multimodal.embed_image')
    @patch('ingestion.embeddings.multimodal.embed_text')
    def test_embed_multi_modal_with_pil_image(self, mock_embed_text, mock_embed_image):
        """Test embedding with PIL Image object."""
        text_emb = np.array([0.1, 0.2, 0.3] * 256)
        image_emb = np.array([0.4, 0.5, 0.6] * 256)
        
        mock_embed_text.return_value = text_emb
        mock_embed_image.return_value = image_emb
        
        mock_pil_image = Image.new('RGB', (100, 100))
        
        result = embed_multi_modal(text="test text", image_path=mock_pil_image, normalize_emb=False)
        
        assert result is not None
        mock_embed_image.assert_called_once_with(mock_pil_image, normalize_emb=False)
    
    def test_embed_multi_modal_no_inputs(self):
        """Test that ValueError is raised when neither text nor image is provided."""
        with pytest.raises(ValueError, match="Must provide either text or image_path"):
            embed_multi_modal()
    
    @patch('ingestion.embeddings.multimodal.embed_text')
    def test_embed_multi_modal_unsupported_image_type(self, mock_embed_text):
        """Test that unsupported image types raise ValueError."""
        text_emb = np.array([0.1, 0.2, 0.3] * 256)
        mock_embed_text.return_value = text_emb
        
        # embed_image will raise ValueError for unsupported types
        with patch('ingestion.embeddings.multimodal.embed_image') as mock_embed_image:
            mock_embed_image.side_effect = ValueError("Unsupported image type: <class 'int'>")
            with pytest.raises(ValueError, match="Unsupported image type"):
                embed_multi_modal(text="test text", image_path=123)

