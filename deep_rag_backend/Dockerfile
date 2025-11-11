FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# System deps for OCR/PDF/graphviz + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr poppler-utils libglib2.0-0 libgl1 graphviz curl git \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Install Python deps (torch/vision already in base image; do NOT reinstall)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY . .
ENV PYTHONPATH=/app

# Graph artifacts directory
RUN mkdir -p /app/inference/graph/artifacts

# entrypoint
COPY scripts/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
