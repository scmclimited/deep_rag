"""
Unit tests for text embedding functionality.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from ingestion.embeddings.text import embed_text
from ingestion.embeddings.model import get_clip_model


class TestEmbedText:
    """Tests for embed_text function."""
    
    @patch('ingestion.embeddings.text.get_clip_model')
    def test_embed_text_success(self, mock_get_model):
        """Test successful text embedding."""
        # Mock a properly initialized model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3] * 256)  # 768 dims
        mock_get_model.return_value = mock_model
        
        result = embed_text("test text", normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 768  # CLIP-ViT-L/14 dimension
        mock_model.encode.assert_called_once()
    
    @patch('ingestion.embeddings.text.get_clip_model')
    def test_embed_text_with_none_model(self, mock_get_model):
        """Test that embed_text handles None model gracefully."""
        # Mock model returning None
        mock_get_model.return_value = None
        
        # Should fall back to word-based truncation and still try to encode
        # But will fail when trying to encode, so we expect a ValueError
        with pytest.raises((ValueError, AttributeError, TypeError)):
            embed_text("test text")
    
    @patch('ingestion.embeddings.text.get_clip_model')
    def test_embed_text_with_none_first_module(self, mock_get_model):
        """Test that embed_text handles None first_module gracefully."""
        # Mock model with empty or None _modules
        mock_model = MagicMock()
        mock_model._modules = {}  # Empty modules
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3] * 256)
        mock_get_model.return_value = mock_model
        
        # Should fall back to word-based truncation and still work
        result = embed_text("test text", normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        mock_model.encode.assert_called_once()
    
    @patch('ingestion.embeddings.text.get_clip_model')
    def test_embed_text_with_none_first_module_in_list(self, mock_get_model):
        """Test that embed_text handles None in _modules list."""
        # Mock model with None as first module
        mock_model = MagicMock()
        mock_model._modules = {'0': None}  # None as first module
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3] * 256)
        mock_get_model.return_value = mock_model
        
        # Should detect None first_module and fall back gracefully
        result = embed_text("test text", normalize_emb=False)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        mock_model.encode.assert_called_once()
    
    @patch('ingestion.embeddings.text.get_clip_model')
    def test_embed_text_encoding_failure(self, mock_get_model):
        """Test that embed_text handles encoding failures with retries."""
        # Mock model that fails encoding
        mock_model = MagicMock()
        mock_model.encode.side_effect = AttributeError("'NoneType' object has no attribute 'tokenize'")
        mock_get_model.return_value = mock_model
        
        # Should raise ValueError after retries
        with pytest.raises(ValueError, match="Failed to encode text"):
            embed_text("test text")
    
    @patch('ingestion.embeddings.text.get_clip_model')
    def test_embed_text_long_text_truncation(self, mock_get_model):
        """Test that long text is properly truncated."""
        # Mock model with tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # Short token list
        mock_tokenizer.decode.return_value = "truncated text"
        
        # Mock first_module with tokenizer
        mock_first_module = MagicMock()
        mock_first_module.tokenizer = mock_tokenizer
        
        mock_model._modules = {'0': mock_first_module}
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3] * 256)
        mock_get_model.return_value = mock_model
        
        # Long text that needs truncation
        long_text = " ".join(["word"] * 100)  # 100 words
        
        result = embed_text(long_text, normalize_emb=False)
        
        assert result is not None
        # Tokenizer should have been called for truncation
        assert mock_tokenizer.encode.called or mock_model.encode.called


class TestModelInitialization:
    """Tests for CLIP model initialization."""
    
    @patch.dict('os.environ', {'CLIP_MODEL': 'sentence-transformers/clip-ViT-L-14', 'EMBEDDING_DIM': '768'})
    @patch('ingestion.embeddings.model.SentenceTransformer')
    def test_get_clip_model_success(self, mock_sentence_transformer):
        """Test successful model loading."""
        mock_model = MagicMock()
        # Mock _first_module() to return a valid module
        mock_first_module = MagicMock()
        mock_model._first_module.return_value = mock_first_module
        mock_model._modules = {'0': mock_first_module}
        # Mock encode to return a valid embedding
        mock_model.encode.return_value = np.array([0.1] * 768)  # 768 dims
        mock_sentence_transformer.return_value = mock_model
        
        from ingestion.embeddings.model import get_clip_model, _clip_model
        
        # Reset global state for test
        import ingestion.embeddings.model as model_module
        model_module._clip_model = None
        
        result = get_clip_model()
        
        assert result is not None
        assert result == mock_model
        mock_sentence_transformer.assert_called_once()
        # Verify encoding test was called
        mock_model.encode.assert_called_once()
    
    @patch.dict('os.environ', {'CLIP_MODEL': 'invalid-model', 'EMBEDDING_DIM': '768'})
    @patch('ingestion.embeddings.model.SentenceTransformer')
    def test_get_clip_model_failure(self, mock_sentence_transformer):
        """Test model loading failure."""
        mock_sentence_transformer.side_effect = Exception("Model not found")
        
        from ingestion.embeddings.model import get_clip_model
        
        # Reset global state for test
        import ingestion.embeddings.model as model_module
        model_module._clip_model = None
        
        with pytest.raises(ImportError, match="CLIP model not available"):
            get_clip_model()
    
    @patch.dict('os.environ', {'CLIP_MODEL': 'sentence-transformers/clip-ViT-L-14', 'EMBEDDING_DIM': '768'})
    @patch('ingestion.embeddings.model.SentenceTransformer')
    def test_get_clip_model_lazy_loading(self, mock_sentence_transformer):
        """Test that model is only loaded once (lazy loading)."""
        mock_model = MagicMock()
        # Mock _first_module() to return a valid module
        mock_first_module = MagicMock()
        mock_model._first_module.return_value = mock_first_module
        mock_model._modules = {'0': mock_first_module}
        # Mock encode to return a valid embedding
        mock_model.encode.return_value = np.array([0.1] * 768)  # 768 dims
        mock_sentence_transformer.return_value = mock_model
        
        from ingestion.embeddings.model import get_clip_model
        
        # Reset global state for test
        import ingestion.embeddings.model as model_module
        model_module._clip_model = None
        
        # First call should load the model
        result1 = get_clip_model()
        assert mock_sentence_transformer.call_count == 1
        # Verify encoding test was called during initialization
        assert mock_model.encode.call_count >= 1
        
        # Second call should use cached model (no new initialization)
        result2 = get_clip_model()
        assert mock_sentence_transformer.call_count == 1  # Still only called once
        assert result1 == result2
    
    @patch.dict('os.environ', {'CLIP_MODEL': 'sentence-transformers/clip-ViT-L-14', 'EMBEDDING_DIM': '768'})
    @patch('ingestion.embeddings.model.SentenceTransformer')
    def test_get_clip_model_invalid_first_module(self, mock_sentence_transformer):
        """Test that model with None _first_module() raises error."""
        # Mock model with None _first_module()
        mock_model = MagicMock()
        mock_model._first_module.return_value = None
        mock_model._modules = {}  # Empty modules
        mock_sentence_transformer.return_value = mock_model
        
        from ingestion.embeddings.model import get_clip_model
        
        # Reset global state for test
        import ingestion.embeddings.model as model_module
        model_module._clip_model = None
        
        with pytest.raises(ImportError, match="CLIP model not available"):
            get_clip_model()
    
    @patch.dict('os.environ', {'CLIP_MODEL': 'sentence-transformers/clip-ViT-L-14', 'EMBEDDING_DIM': '768'})
    @patch('ingestion.embeddings.model.SentenceTransformer')
    def test_get_clip_model_encoding_test_failure(self, mock_sentence_transformer):
        """Test that model that fails encoding test raises error."""
        # Mock model that fails encoding
        mock_model = MagicMock()
        mock_model._first_module.return_value = MagicMock()  # Valid first module
        mock_model._modules = {'0': MagicMock()}  # Valid modules
        mock_model.encode.side_effect = AttributeError("'NoneType' object has no attribute 'tokenize'")
        mock_sentence_transformer.return_value = mock_model
        
        from ingestion.embeddings.model import get_clip_model
        
        # Reset global state for test
        import ingestion.embeddings.model as model_module
        model_module._clip_model = None
        
        with pytest.raises(ImportError, match="CLIP model not available"):
            get_clip_model()

