"""
Unit tests for retrieval.sanitize module.
"""
import pytest
from retrieval.sanitize import sanitize_query_for_tsquery


def test_sanitize_basic():
    """Test basic query sanitization."""
    query = "test query"
    result = sanitize_query_for_tsquery(query)
    assert result == "test query"


def test_sanitize_ampersand():
    """Test ampersand replacement."""
    query = "test & query"
    result = sanitize_query_for_tsquery(query)
    assert result == "test and query"


def test_sanitize_special_chars():
    """Test special character removal."""
    query = "test! query| with: special* chars"
    result = sanitize_query_for_tsquery(query)
    assert "!" not in result
    assert "|" not in result
    assert ":" not in result
    assert "*" not in result


def test_sanitize_quotes():
    """Test quote removal."""
    query = 'test "query" with quotes'
    result = sanitize_query_for_tsquery(query)
    assert '"' not in result
    assert "'" not in result


def test_sanitize_bullet_points():
    """Test bullet point removal."""
    query = "• test query"
    result = sanitize_query_for_tsquery(query)
    assert not result.startswith("•")


def test_sanitize_whitespace():
    """Test whitespace normalization."""
    query = "test    query   with   spaces"
    result = sanitize_query_for_tsquery(query)
    assert "  " not in result  # No double spaces
