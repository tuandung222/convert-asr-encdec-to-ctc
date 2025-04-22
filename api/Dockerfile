# Stage 1: Build dependencies
FROM python:3.10.12-slim AS builder

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_DEFAULT_TIMEOUT=100
ARG PIP_NO_CACHE_DIR=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# First install numpy 1.x explicitly (for compatibility)
RUN pip install --no-cache-dir 'numpy<2.0.0'

# Then install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch==2.2.1+cpu \
    torchaudio==2.2.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy and install other requirements
COPY api/requirements.txt .
# Make sure numpy doesn't get upgraded to 2.x
RUN grep -v numpy requirements.txt > requirements_no_numpy.txt && \
    pip install --no-cache-dir -r requirements_no_numpy.txt

# Stage 2: Runtime
FROM python:3.10.12-slim

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG APP_USER=api
ARG APP_USER_UID=1000

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/checkpoints /tmp/prometheus-metrics
RUN useradd -m -u ${APP_USER_UID} ${APP_USER}
RUN chown -R ${APP_USER}:${APP_USER} /app /tmp/prometheus-metrics

# Copy only necessary application files
COPY src/models /app/src/models
COPY src/utils /app/src/utils
COPY api/metrics.py /app/src/
COPY api /app/api

# Set ownership of copied files
RUN chown -R ${APP_USER}:${APP_USER} /app

# Switch to non-root user
USER ${APP_USER}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    INFERENCE_DEVICE=cpu \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus-metrics \
    # Tracing configuration
    SERVICE_NAME=asr-api \
    ENABLE_TRACING=true \
    JAEGER_HOST=jaeger \
    JAEGER_PORT=6831 \
    # Don't attempt to use OpenTelemetry middleware
    SKIP_OTEL_MIDDLEWARE=true

# # Install numba
# RUN pip install --no-cache-dir numba

# # Install soundfile
# RUN pip install --no-cache-dir soundfile

# # Install resampy
# RUN pip install --no-cache-dir resampy

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=200s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
