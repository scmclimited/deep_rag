"""
Pytest configuration and shared fixtures for all tests.
"""
import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables
os.environ.setdefault("DB_HOST")
os.environ.setdefault("DB_PORT")
os.environ.setdefault("DB_USER")
os.environ.setdefault("DB_PASS")
os.environ.setdefault("DB_NAME")

@pytest.fixture(scope="session")
def test_env():
    """Fixture to provide test environment variables."""
    return {
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_PORT": os.getenv("DB_PORT"),
        "DB_USER": os.getenv("DB_USER"),
        "DB_PASS": os.getenv("DB_PASS"),
        "DB_NAME": os.getenv("DB_NAME"),
    }

