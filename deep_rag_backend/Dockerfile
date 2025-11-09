FROM python:3.11-slim

# System deps for OCR and PDF rendering
RUN apt-get update && apt-get install -y \
    tesseract-ocr poppler-utils libglib2.0-0 libgl1 graphviz curl \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --upgrade pip

WORKDIR /app

# Python deps - optimized for caching with CUDA PyTorch
# Use BuildKit cache mount for pip cache (faster rebuilds)
# Install CUDA PyTorch first (required for torchvision operators like nms)
# Then install other requirements (pip will use existing torch if version matches)
# Note: Requires Docker BuildKit (enabled by default in Docker Desktop)
# Note: CUDA PyTorch is required because torchvision::nms operator needs CUDA support
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch>=2.0.0 torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# App code (copy all backend code, excluding test logs via .dockerignore)
COPY . .

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH=/app

# Create artifacts directory for graph visualization
RUN mkdir -p /app/inference/graph/artifacts

# Copy entrypoint script to root and ensure it has Unix line endings
# This must be done after COPY . . to ensure the script exists
COPY scripts/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh && \
    echo "Entrypoint script copied and configured successfully"

EXPOSE 8000

# Use entrypoint script (can run tests on startup if RUN_TESTS_ON_STARTUP=true)
ENTRYPOINT ["/entrypoint.sh"]
