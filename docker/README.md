# Docker Deployment Guide

This directory contains modular Docker Compose files for flexible deployment options of the Vietnamese ASR system.

## Structure

- `docker-compose.base.yml` - Contains common networks and volumes
- `docker-compose.api.yml` - ASR API service
- `docker-compose.ui.yml` - Streamlit UI service
- `docker-compose.monitoring.yml` - Monitoring stack (Prometheus, Grafana, Jaeger, etc.)
- `docker-compose.full.yml` - Reference file for full deployment

## Usage

### Running the Full Stack

To run the entire application stack:

```bash
docker-compose -f docker/docker-compose.base.yml \
               -f docker/docker-compose.api.yml \
               -f docker/docker-compose.ui.yml \
               -f docker/docker-compose.monitoring.yml up -d
```

### Running Only the API

```bash
docker-compose -f docker/docker-compose.base.yml \
               -f docker/docker-compose.api.yml up -d
```

### Running with Monitoring Stack

```bash
docker-compose -f docker/docker-compose.base.yml \
               -f docker/docker-compose.api.yml \
               -f docker/docker-compose.monitoring.yml up -d
```

### Local Testing Setup

```bash
docker-compose -f docker/docker-compose.base.yml \
               -f docker/docker-compose.monitoring.yml up -d
```

Then run the API and UI services directly on the host machine:

```bash
# Terminal 1 - Run API
cd api
uvicorn app:app --host 0.0.0.0 --port 8000

# Terminal 2 - Run UI
cd ui
streamlit run app.py
```

## Development Workflow

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency. To set up:

1. Install pre-commit:
   ```bash
   pip install pre-commit
   ```

2. Install the git hooks:
   ```bash
   pre-commit install
   ```

3. Run pre-commit on all files:
   ```bash
   pre-commit run --all-files
   ```

The hooks will automatically run on each commit. They include:
- Code formatting (Black, isort)
- Linting (Flake8)
- Type checking (MyPy)
- Dockerfile linting (hadolint)
- Security checks (Bandit)
- And more

## Configuration

Environment variables can be customized by creating a `.env` file in the project root directory.

Example:
```
# API Configuration
PORT=8000
INFERENCE_DEVICE=cpu
ENVIRONMENT=production

# UI Configuration
API_URL=http://api:8000
GRAFANA_URL=http://grafana:3000
```

## Accessing Services

- ASR API: http://localhost:8000
- Streamlit UI: http://localhost:8501
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (username: admin, password: F7aJw3kQ9pL5xYzR)
- Jaeger: http://localhost:16686

## Resource Requirements

Each service has resource constraints configured through the `deploy` section:

- API: 1 CPU, 1GB RAM
- UI: 0.5 CPU, 512MB RAM
- Monitoring stack: Variable based on service

## Production Deployment

For production environments, consider:

1. Setting stronger passwords for Grafana in a secrets file
2. Enabling SSL/TLS for all services
3. Using a container orchestration platform like Kubernetes
4. Setting up proper backup for Prometheus and Grafana data volumes 