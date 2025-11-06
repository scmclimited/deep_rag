FROM python:3.11-slim

# System deps for OCR and PDF rendering
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y \
    tesseract-ocr poppler-utils libglib2.0-0 libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "inference.service:app", "--host", "0.0.0.0", "--port", "8000"]
