# Vietnamese ASR with MLOps Architecture

## Overview

This document outlines the MLOps architecture for the Vietnamese ASR system using PhoWhisper-CTC. The architecture incorporates DevOps best practices and modern cloud-native technologies.

## Components

### 1. API Service (FastAPI)

- **Purpose**: Provides REST endpoints for speech-to-text transcription
- **Technology**: FastAPI for high-performance API development
- **Endpoints**:
  - `/transcribe` - Accepts audio files for transcription
  - `/health` - Health check endpoint
  - `/metrics` - Prometheus metrics endpoint

### 2. Model Service

- **Purpose**: Handles model inference using our PhoWhisper-CTC model
- **Technology**: PyTorch with CPU optimization
- **Features**:
  - Model caching for improved performance
  - Batch processing support
  - OpenTelemetry instrumentation for tracing

### 3. Containerization

- **Docker**: Individual containers for each service
- **Docker Compose**: Local development and testing environment
- **Images**:
  - `asr-api`: FastAPI service
  - `asr-model`: Model inference service
  - Supporting services (Prometheus, Grafana, Jaeger)

### 4. Monitoring and Observability

- **Prometheus**: Metrics collection
  - API request rates, latencies, errors
  - Model inference times, batch sizes
  - Resource utilization metrics
- **Grafana**: Visualization dashboards
  - Real-time performance monitoring
  - Resource utilization graphs
  - SLA/SLO tracking
- **Jaeger & OpenTelemetry**: Distributed tracing
  - End-to-end request flows
  - Performance bottleneck identification

### 5. CI/CD Pipeline

- **Jenkins**: Automated build and deployment pipeline
  - Test automation
  - Docker image building
  - Version management
  - Deployment to Kubernetes

### 6. Kubernetes Deployment

- **Configuration**: 
  - 3 API service replicas
  - Autoscaling based on CPU/memory utilization
  - Rolling updates for zero-downtime deployments
- **Helm Charts**:
  - Custom charts for ASR services
  - Prometheus and Grafana deployment

### 7. Cloud Integration (Optional)

- **Potential providers**: GCP, AWS, or Azure
- **Services**:
  - Managed Kubernetes (GKE, EKS, or AKS)
  - Object storage for model weights and audio files
  - Container Registry for Docker images
  - CI/CD integration

## Architecture Diagram

```
┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │
│  Client         │──────▶  FastAPI        │
│  (Applications) │      │  Service        │
│                 │      │                 │
└─────────────────┘      └─────────┬───────┘
                                   │
                                   ▼
┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │
│  Prometheus     │◀─────│  Model          │
│  Monitoring     │      │  Service        │
│                 │      │                 │
└─────────┬───────┘      └─────────────────┘
          │
          ▼
┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │
│  Grafana        │      │  Jaeger         │
│  Dashboards     │      │  Tracing        │
│                 │      │                 │
└─────────────────┘      └─────────────────┘
```

## Implementation Roadmap

1. FastAPI Service Development
2. Docker Containerization
3. Docker Compose Setup
4. Monitoring Integration (Prometheus + Grafana)
5. OpenTelemetry Tracing with Jaeger
6. Jenkins CI/CD Pipeline
7. Kubernetes Deployment
8. Cloud Integration

## Performance Considerations

- Model loading time optimization
- Batch processing for higher throughput
- Caching strategies for improved latency
- Resource allocation tuning in Kubernetes
- Horizontal scaling based on traffic patterns 