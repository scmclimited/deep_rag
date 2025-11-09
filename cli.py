"""
Root CLI wrapper - Delegates to backend CLI with proper path handling.
This allows the deep-rag command to work from the project root.

When installed via `pip install -e .`, this module delegates to the backend CLI
located in `deep_rag_backend/inference/cli.py`, ensuring all imports work correctly.
"""
import sys
from pathlib import Path

# Get the root directory (where this file is located)
root_dir = Path(__file__).parent.resolve()
backend_dir = root_dir / "deep_rag_backend"

# Verify backend directory exists
if not backend_dir.exists():
    raise ImportError(
        f"Backend directory not found: {backend_dir}\n"
        f"Expected structure: {root_dir}/deep_rag_backend/inference/cli.py"
    )

# Add backend directory to Python path so imports work correctly
# By adding deep_rag_backend to sys.path, Python will look there for modules.
# This means "from inference.cli import app" will find:
#   deep_rag_backend/inference/cli.py
# instead of looking for inference in the project root.
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Import and delegate to backend CLI
# This import works because deep_rag_backend is now in sys.path.
# Python will find: deep_rag_backend/inference/cli.py
# The backend CLI will handle all commands (ingest, query, infer, etc.)
from inference.cli import app

if __name__ == "__main__":
    app()

