FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# System deps for OCR/PDF/graphviz + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr poppler-utils libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --upgrade pip  \
 && pip install graphviz && pip install --upgrade sentence-transformers

WORKDIR /app

# Install Python deps (torch/vision already in base image; do NOT reinstall)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Create models directory (will be empty if no models pre-downloaded)
RUN mkdir -p /app/models

# Copy models directory if it exists (pre-downloaded models)
# This allows models to be baked into the image instead of downloading at runtime
# To use pre-downloaded models, run: python scripts/download_model.py before building
# Then models will be copied into the image
# Note: If models/ doesn't exist, create an empty .gitkeep file to allow COPY to work
COPY models/ ./models/

# App code
COPY . .
ENV PYTHONPATH=/app

# Set default Hugging Face cache directory (can be overridden via env var)
# Using HF_HOME instead of deprecated TRANSFORMERS_CACHE (for transformers v5+ compatibility)
ENV HF_HOME=/app/.cache/huggingface

# Graph artifacts directory
RUN mkdir -p /app/inference/graph/artifacts

# entrypoint
COPY scripts/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
