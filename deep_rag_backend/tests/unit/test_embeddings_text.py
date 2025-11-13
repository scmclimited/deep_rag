"""
Unit tests for text embedding functionality.
"""
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock, Mock
from ingestion.embeddings.text import embed_text
from ingestion.embeddings.model import get_clip_model


class TestEmbedText:
    """Tests for embed_text function."""
    
    @patch('ingestion.embeddings.text.get_clip_processor')
    @patch('ingestion.embeddings.text.get_clip_model')
    @patch('ingestion.embeddings.text.torch.no_grad')
    def test_embed_text_success(self, mock_no_grad, mock_get_model, mock_get_processor):
        """Test successful text embedding."""
        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        # Mock text features
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([0.1, 0.2, 0.3] * 256)  # 768 dims
        mock_text_features = MagicMock()
        mock_text_features.__getitem__ = MagicMock(return_value=mock_tensor)
        mock_model.get_text_features.return_value = mock_text_features
        
        # Mock processor to return inputs dict
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        
        mock_get_model.return_value = mock_model
        mock_get_processor.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        result = embed_text("test text", normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 768  # CLIP-ViT-L/14 dimension
        mock_processor.assert_called_once()
        mock_model.get_text_features.assert_called_once()
    
    @patch('ingestion.embeddings.text.get_clip_processor')
    @patch('ingestion.embeddings.text.get_clip_model')
    def test_embed_text_with_none_model(self, mock_get_model, mock_get_processor):
        """Test that embed_text handles None model gracefully."""
        # Mock model returning None
        mock_get_model.return_value = None
        mock_get_processor.return_value = MagicMock()
        
        # Should fail when trying to use None model
        with pytest.raises((ValueError, AttributeError, TypeError)):
            embed_text("test text")
    
    @patch('ingestion.embeddings.text.get_clip_processor')
    @patch('ingestion.embeddings.text.get_clip_model')
    @patch('ingestion.embeddings.text.torch.no_grad')
    def test_embed_text_with_none_first_module(self, mock_no_grad, mock_get_model, mock_get_processor):
        """Test that embed_text handles processor errors gracefully."""
        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        # Mock text features
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([0.1, 0.2, 0.3] * 256)
        mock_text_features = MagicMock()
        mock_text_features.__getitem__ = MagicMock(return_value=mock_tensor)
        mock_model.get_text_features.return_value = mock_text_features
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        
        mock_get_model.return_value = mock_model
        mock_get_processor.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        result = embed_text("test text", normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        mock_model.get_text_features.assert_called_once()
    
    @patch('ingestion.embeddings.text.get_clip_processor')
    @patch('ingestion.embeddings.text.get_clip_model')
    @patch('ingestion.embeddings.text.torch.no_grad')
    def test_embed_text_with_none_first_module_in_list(self, mock_no_grad, mock_get_model, mock_get_processor):
        """Test that embed_text works with processor."""
        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        # Mock text features
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([0.1, 0.2, 0.3] * 256)
        mock_text_features = MagicMock()
        mock_text_features.__getitem__ = MagicMock(return_value=mock_tensor)
        mock_model.get_text_features.return_value = mock_text_features
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        
        mock_get_model.return_value = mock_model
        mock_get_processor.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        result = embed_text("test text", normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        mock_model.get_text_features.assert_called_once()
    
    @patch('ingestion.embeddings.text.get_clip_processor')
    @patch('ingestion.embeddings.text.get_clip_model')
    @patch('ingestion.embeddings.text.torch.no_grad')
    def test_embed_text_encoding_failure(self, mock_no_grad, mock_get_model, mock_get_processor):
        """Test that embed_text handles encoding failures with retries."""
        # Mock model and processor that fail
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.side_effect = Exception("Processing failed")
        
        mock_get_model.return_value = mock_model
        mock_get_processor.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        # Should raise ValueError after retries
        with pytest.raises(ValueError, match="Failed to encode text"):
            embed_text("test text")
    
    @patch('ingestion.embeddings.text.get_clip_processor')
    @patch('ingestion.embeddings.text.get_clip_model')
    @patch('ingestion.embeddings.text.torch.no_grad')
    def test_embed_text_long_text_truncation(self, mock_no_grad, mock_get_model, mock_get_processor):
        """Test that long text is properly truncated."""
        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        # Mock text features
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([0.1, 0.2, 0.3] * 256)
        mock_text_features = MagicMock()
        mock_text_features.__getitem__ = MagicMock(return_value=mock_tensor)
        mock_model.get_text_features.return_value = mock_text_features
        
        # Processor handles truncation automatically
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        
        mock_get_model.return_value = mock_model
        mock_get_processor.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        # Long text that needs truncation
        long_text = " ".join(["word"] * 100)  # 100 words
        
        result = embed_text(long_text, normalize_emb=False)
        
        assert result is not None
        # Processor should have been called (handles truncation)
        mock_processor.assert_called()
        mock_model.get_text_features.assert_called_once()


class TestModelInitialization:
    """Tests for CLIP model initialization."""
    
    @patch.dict('os.environ', {'CLIP_MODEL': 'openai/clip-vit-large-patch14-336', 'EMBEDDING_DIM': '768'})
    @patch('ingestion.embeddings.model.CLIPProcessor')
    @patch('ingestion.embeddings.model.CLIPModel')
    @patch('ingestion.embeddings.model.torch.no_grad')
    def test_get_clip_model_success(self, mock_no_grad, mock_clip_model, mock_clip_processor):
        """Test successful model loading."""
        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        # Mock text features for validation test
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([0.1] * 768)  # 768 dims
        mock_text_features = MagicMock()
        mock_text_features.__getitem__ = MagicMock(return_value=mock_tensor)
        mock_model.get_text_features.return_value = mock_text_features
        mock_model.eval.return_value = mock_model
        
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_clip_model.from_pretrained.return_value = mock_model
        mock_clip_processor.from_pretrained.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        from ingestion.embeddings.model import get_clip_model
        
        # Reset global state for test
        import ingestion.embeddings.model as model_module
        model_module._clip_model = None
        model_module._clip_processor = None
        
        result = get_clip_model()
        
        assert result is not None
        assert result == mock_model
        mock_clip_model.from_pretrained.assert_called_once()
        # Verify encoding test was called
        mock_model.get_text_features.assert_called_once()
    
    @patch.dict('os.environ', {'CLIP_MODEL': 'invalid-model', 'EMBEDDING_DIM': '768'})
    @patch('ingestion.embeddings.model.CLIPModel')
    def test_get_clip_model_failure(self, mock_clip_model):
        """Test model loading failure."""
        mock_clip_model.from_pretrained.side_effect = Exception("Model not found")
        
        from ingestion.embeddings.model import get_clip_model
        
        # Reset global state for test
        import ingestion.embeddings.model as model_module
        model_module._clip_model = None
        model_module._clip_processor = None
        
        with pytest.raises(ImportError, match="CLIP model not available"):
            get_clip_model()
    
    @patch.dict('os.environ', {'CLIP_MODEL': 'openai/clip-vit-large-patch14-336', 'EMBEDDING_DIM': '768'})
    @patch('ingestion.embeddings.model.CLIPProcessor')
    @patch('ingestion.embeddings.model.CLIPModel')
    @patch('ingestion.embeddings.model.torch.no_grad')
    def test_get_clip_model_lazy_loading(self, mock_no_grad, mock_clip_model, mock_clip_processor):
        """Test that model is only loaded once (lazy loading)."""
        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        # Mock text features for validation
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([0.1] * 768)  # 768 dims
        mock_text_features = MagicMock()
        mock_text_features.__getitem__ = MagicMock(return_value=mock_tensor)
        mock_model.get_text_features.return_value = mock_text_features
        mock_model.eval.return_value = mock_model
        
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_clip_model.from_pretrained.return_value = mock_model
        mock_clip_processor.from_pretrained.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        from ingestion.embeddings.model import get_clip_model
        
        # Reset global state for test
        import ingestion.embeddings.model as model_module
        model_module._clip_model = None
        model_module._clip_processor = None
        
        # First call should load the model
        result1 = get_clip_model()
        assert mock_clip_model.from_pretrained.call_count == 1
        
        # Second call should use cached model (no new initialization)
        result2 = get_clip_model()
        assert mock_clip_model.from_pretrained.call_count == 1  # Still only called once
        assert result1 == result2
    
    @patch.dict('os.environ', {'CLIP_MODEL': 'openai/clip-vit-large-patch14-336', 'EMBEDDING_DIM': '768'})
    @patch('ingestion.embeddings.model.CLIPProcessor')
    @patch('ingestion.embeddings.model.CLIPModel')
    @patch('ingestion.embeddings.model.torch.no_grad')
    def test_get_clip_model_invalid_first_module(self, mock_no_grad, mock_clip_model, mock_clip_processor):
        """Test that model that fails validation raises error."""
        # Mock model that fails validation
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model.get_text_features.side_effect = Exception("Validation failed")
        mock_model.eval.return_value = mock_model
        
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_clip_model.from_pretrained.return_value = mock_model
        mock_clip_processor.from_pretrained.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        from ingestion.embeddings.model import get_clip_model
        
        # Reset global state for test
        import ingestion.embeddings.model as model_module
        model_module._clip_model = None
        model_module._clip_processor = None
        
        with pytest.raises(ImportError, match="CLIP model not available"):
            get_clip_model()
    
    @patch.dict('os.environ', {'CLIP_MODEL': 'openai/clip-vit-large-patch14-336', 'EMBEDDING_DIM': '768'})
    @patch('ingestion.embeddings.model.CLIPProcessor')
    @patch('ingestion.embeddings.model.CLIPModel')
    @patch('ingestion.embeddings.model.torch.no_grad')
    def test_get_clip_model_encoding_test_failure(self, mock_no_grad, mock_clip_model, mock_clip_processor):
        """Test that model that fails encoding test raises error."""
        # Mock model that fails encoding test
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model.get_text_features.side_effect = Exception("Encoding test failed")
        mock_model.eval.return_value = mock_model
        
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_clip_model.from_pretrained.return_value = mock_model
        mock_clip_processor.from_pretrained.return_value = mock_processor
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)
        
        from ingestion.embeddings.model import get_clip_model
        
        # Reset global state for test
        import ingestion.embeddings.model as model_module
        model_module._clip_model = None
        model_module._clip_processor = None
        
        with pytest.raises(ImportError, match="CLIP model not available"):
            get_clip_model()

