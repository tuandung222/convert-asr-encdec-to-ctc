FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install additional API dependencies
RUN pip install --no-cache-dir fastapi uvicorn python-multipart opentelemetry-api \
    opentelemetry-sdk opentelemetry-exporter-jaeger \
    opentelemetry-instrumentation-fastapi prometheus-fastapi-instrumentator

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    INFERENCE_DEVICE=cpu \
    MODEL_CONFIG_PATH=/app/configs/model_config.yaml \
    INFERENCE_CONFIG_PATH=/app/configs/inference_config.yaml \
    ENVIRONMENT=production

# Expose the application port
EXPOSE 8000

# Create a non-root user
RUN useradd -m appuser
USER appuser

# Run the application
CMD ["python", "api/main.py"] 