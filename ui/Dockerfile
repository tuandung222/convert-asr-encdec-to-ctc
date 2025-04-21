# Stage 1: Build dependencies
FROM python:3.10.12-slim as builder

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_DEFAULT_TIMEOUT=100
ARG PIP_NO_CACHE_DIR=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10.12-slim

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG APP_USER=streamlit
ARG APP_USER_UID=1000

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user
RUN useradd -m -u ${APP_USER_UID} ${APP_USER}

# Copy all UI files
COPY ui/app.py .
COPY ui/utils.py .
COPY ui/components.py .
COPY ui/static ./static

# Set proper permissions
RUN chown -R ${APP_USER}:${APP_USER} /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_URL=http://api:8000 \
    GRAFANA_URL=http://grafana:3000 \
    JAEGER_URL=http://jaeger:16686 \
    ENABLE_TRACE_PROPAGATION=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Switch to non-root user
USER ${APP_USER}

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py"]
