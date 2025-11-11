#!/bin/bash
# Entrypoint script for Docker container
# Can optionally run tests before starting the API server

set -e

# If RUN_TESTS_ON_STARTUP is set, run tests first
if [ "${RUN_TESTS_ON_STARTUP:-false}" = "true" ]; then
    echo "=========================================="
    echo "Running database schema tests on startup..."
    echo "=========================================="
    python -m pytest tests/integration/test_database_schema.py -v || {
        echo "WARNING: Database schema tests failed. Continuing anyway..."
    }
    echo "=========================================="
    echo "Tests completed. Starting API server..."
    echo "=========================================="
fi

# === GPU / CUDA Sanity Check ===
echo "=========================================="
echo "Verifying CUDA and PyTorch environment..."
echo "=========================================="

python - <<'PY'
import torch, sys
print("torch:", torch.__version__)
try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except ImportError:
    print("torchvision: not installed")
available = torch.cuda.is_available()
print("cuda available:", available)
if available:
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA not available in this container. Continuing anyway...")
    print("         The application will run but may be slower without GPU acceleration.")
PY

echo "=========================================="
echo "GPU verification complete. Starting API..."
echo "=========================================="

# Start the API server with verbose logging
exec uvicorn inference.service:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --access-log \
    --no-use-colors \
    --workers ${UVICORN_WORKERS:-8}
