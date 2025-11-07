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

# Start the API server
exec uvicorn inference.service:app --host 0.0.0.0 --port 8000

