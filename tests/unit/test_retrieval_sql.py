"""
Unit tests for retrieval.sql module.
"""
import pytest
from retrieval.sql import get_hybrid_sql, get_hybrid_sql_with_exclusion


def test_get_hybrid_sql_without_doc_id():
    """Test SQL generation without doc_id filter."""
    sql = get_hybrid_sql(embedding_dim=768, doc_id=None)
    
    assert "WITH" in sql
    assert "q AS" in sql
    assert "lex AS" in sql
    assert "vec AS" in sql
    assert "vector(768)" in sql
    assert "%(doc_id)s" not in sql  # No doc_id filter


def test_get_hybrid_sql_with_doc_id():
    """Test SQL generation with doc_id filter."""
    sql = get_hybrid_sql(embedding_dim=768, doc_id="test-doc-id")
    
    assert "WITH" in sql
    assert "q AS" in sql
    assert "lex AS" in sql
    assert "vec AS" in sql
    assert "vector(768)" in sql
    assert "%(doc_id)s" in sql  # Has doc_id filter
    assert "AND c.doc_id = %(doc_id)s" in sql


def test_get_hybrid_sql_with_exclusion():
    """Test SQL generation with exclusion filter."""
    sql = get_hybrid_sql_with_exclusion(embedding_dim=768, exclude_doc_id="test-doc-id")
    
    assert "WITH" in sql
    assert "q AS" in sql
    assert "lex AS" in sql
    assert "vec AS" in sql
    assert "vector(768)" in sql
    assert "%(exclude_doc_id)s" in sql  # Has exclusion filter
    assert "AND c.doc_id != %(exclude_doc_id)s" in sql


def test_get_hybrid_sql_different_dimensions():
    """Test SQL generation with different embedding dimensions."""
    sql_768 = get_hybrid_sql(embedding_dim=768, doc_id=None)
    sql_512 = get_hybrid_sql(embedding_dim=512, doc_id=None)
    
    assert "vector(768)" in sql_768
    assert "vector(512)" in sql_512
