"""
Integration tests for thread tracking functionality.
"""
import pytest
import os
from dotenv import load_dotenv
from retrieval.thread_tracking.log import log_thread_interaction
from retrieval.thread_tracking.get import get_thread_interactions

load_dotenv()


class TestThreadTracking:
    """Tests for thread tracking functionality."""
    
    @pytest.mark.skipif(
        not all([
            os.getenv("DB_HOST"),
            os.getenv("DB_USER"),
            os.getenv("DB_PASS"),
            os.getenv("DB_NAME")
        ]),
        reason="Database connection environment variables not set"
    )
    def test_log_thread_interaction(self):
        """Test logging a thread interaction."""
        try:
            record_id = log_thread_interaction(
                user_id="test_user",
                thread_id="test_thread",
                query_text="What is the answer?",
                doc_ids=["doc1", "doc2"],
                final_answer="This is the answer.",
                entry_point="cli",
                pipeline_type="langgraph",
                cross_doc=False,
                metadata={"confidence": 0.85, "iterations": 1}
            )
            
            assert record_id is not None
            assert isinstance(record_id, int)
            print(f"✅ Logged thread interaction with ID: {record_id}")
            
        except Exception as e:
            pytest.fail(f"Failed to log thread interaction: {e}")
    
    @pytest.mark.skipif(
        not all([
            os.getenv("DB_HOST"),
            os.getenv("DB_USER"),
            os.getenv("DB_PASS"),
            os.getenv("DB_NAME")
        ]),
        reason="Database connection environment variables not set"
    )
    def test_get_thread_interactions(self):
        """Test retrieving thread interactions."""
        try:
            # First log an interaction
            log_thread_interaction(
                user_id="test_user",
                thread_id="test_thread_retrieve",
                query_text="Test question",
                doc_ids=["doc1"],
                final_answer="Test answer",
                entry_point="cli",
                pipeline_type="langgraph"
            )
            
            # Then retrieve it
            interactions = get_thread_interactions(thread_id="test_thread_retrieve", limit=10)
            
            assert interactions is not None
            assert len(interactions) > 0
            assert interactions[0]["thread_id"] == "test_thread_retrieve"
            print(f"✅ Retrieved {len(interactions)} thread interaction(s)")
            
        except Exception as e:
            pytest.fail(f"Failed to retrieve thread interactions: {e}")

