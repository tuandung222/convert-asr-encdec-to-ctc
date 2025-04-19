FROM python:3.10-slim

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

# Copy model code and API code
COPY src /app/src
COPY api /app/api
COPY scripts /app/scripts
COPY checkpoints /app/checkpoints

# Create directory for checkpoints
RUN mkdir -p /app/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    INFERENCE_DEVICE=cpu \
    ENVIRONMENT=production \
    MODEL_CHECKPOINT_PATH=/app/checkpoints/best-val_wer=0.3986.ckpt

# Expose the application port
EXPOSE 8000

# Run the application with uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"] 