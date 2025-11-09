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

# Set environment variable to indicate test environment
# This ensures AgentLogger uses test logs directory
os.environ["PYTEST_CURRENT_TEST"] = "1"

# Create test logs directory if it doesn't exist
test_logs_dir = project_root / "inference" / "graph" / "logs" / "test"
test_logs_dir.mkdir(parents=True, exist_ok=True)

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

