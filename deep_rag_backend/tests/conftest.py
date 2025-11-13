"""
Pytest configuration and shared fixtures for all tests.
"""
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variable to indicate test environment
# This ensures AgentLogger uses test logs directory
os.environ["PYTEST_CURRENT_TEST"] = "1"

# Create test logs directory if it doesn't exist
test_logs_dir = project_root / "inference" / "graph" / "logs" / "test"
test_logs_dir.mkdir(parents=True, exist_ok=True)

# Mock google.genai if not available (for test environments without google-genai installed)
# This allows tests to run locally even if the package isn't installed
# Docker builds will use the real package
try:
    from google import genai
    from google.genai import types
except ImportError:
    # Create mock module structure silently (no warnings needed)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mock_genai = MagicMock()
        mock_genai.Client = MagicMock()
        
        # Create mock types module
        mock_types = MagicMock()
        mock_types.GenerateContentConfig = MagicMock
        mock_types.HttpOptions = MagicMock
        
        # Inject into sys.modules so imports work
        sys.modules['google'] = MagicMock(genai=mock_genai)
        sys.modules['google.genai'] = mock_genai
        sys.modules['google.genai.types'] = mock_types

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

