# DEPRECATED - DO NOT USE
#
# This file has been replaced by modular Dockerfiles in each service directory:
# - api/Dockerfile
# - ui/Dockerfile
#
# Please see the docker/ directory for the new Docker Compose configuration

# FROM python:3.10-slim-bullseye

# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     ffmpeg \
#     libsndfile1 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements first for better caching
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PORT=8000 \
#     INFERENCE_DEVICE=cpu

# # Expose the port the app runs on
# EXPOSE 8000

# # Command to run the application
# CMD ["python", "-m", "api.app"]
