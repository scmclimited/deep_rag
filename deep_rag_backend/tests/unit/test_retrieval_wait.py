"""
Unit tests for retrieval wait_for_chunks function.
"""
import pytest
import uuid
from unittest.mock import patch, MagicMock
from retrieval.retrieval import wait_for_chunks


class TestWaitForChunks:
    """Tests for wait_for_chunks function."""
    
    @patch('retrieval.wait.connect')
    def test_wait_for_chunks_success(self, mock_connect):
        """Test successful chunk waiting."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = [5]  # 5 chunks found
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Use a real UUID format for testing
        test_doc_id = str(uuid.uuid4())
        result = wait_for_chunks(test_doc_id, max_wait_seconds=1)
        assert result == 5
    
    @patch('retrieval.wait.connect')
    def test_wait_for_chunks_timeout(self, mock_connect):
        """Test timeout when chunks not found."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = [0]  # No chunks found
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Use a real UUID format for testing
        test_doc_id = str(uuid.uuid4())
        with pytest.raises(TimeoutError):
            wait_for_chunks(test_doc_id, max_wait_seconds=0.1)

