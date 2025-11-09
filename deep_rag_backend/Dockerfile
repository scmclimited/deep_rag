FROM python:3.11-slim

# System deps for OCR and PDF rendering
RUN apt-get update && apt-get install -y \
    tesseract-ocr poppler-utils libglib2.0-0 libgl1 graphviz curl \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --upgrade pip

WORKDIR /app

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code (copy all backend code, excluding test logs via .dockerignore)
COPY . .

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH=/app

# Create artifacts directory
RUN mkdir -p /app/artifacts

# Copy entrypoint script to root and ensure it has Unix line endings
# This must be done after COPY . . to ensure the script exists
COPY scripts/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh && \
    echo "Entrypoint script copied and configured successfully"

EXPOSE 8000

# Use entrypoint script (can run tests on startup if RUN_TESTS_ON_STARTUP=true)
ENTRYPOINT ["/entrypoint.sh"]
