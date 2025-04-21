# Vietnamese Speech Recognition with PhoWhisper-CTC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Pulls](https://img.shields.io/docker/pulls/tuandung12092002/asr-fastapi-server)](https://hub.docker.com/r/tuandung12092002/asr-fastapi-server)
[![CI/CD](https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc/actions/workflows/docker-publish.yml)

A high-performance Vietnamese Automatic Speech Recognition (ASR) system using a CTC-based architecture derived from PhoWhisper. This project implements a complete ML system with FastAPI backend, interactive UI, observability, and cloud deployment options.

## 🚀 Features

- **Fast and Accurate** Vietnamese speech recognition (2-3x faster than real-time)
- **CTC-based architecture** for efficient inference and simplified training
- **Multiple deployment options**:
  - **Docker Compose** for local and single-server deployment
  - **Kubernetes** for scalable cloud deployment on Digital Ocean
- **Comprehensive monitoring**:
  - Prometheus metrics
  - Grafana dashboards
  - Jaeger distributed tracing
- **Modern UI options**:
  - Streamlit web interface
  - Gradio demo
- **CI/CD pipeline** with Jenkins and GitHub Actions
- **ONNX optimization** with INT8 quantization for CPU

## 📊 System Architecture

```mermaid
graph TD
    User[User]-->|Upload/Record Audio|UI[Streamlit/Gradio UI]
    UI-->|HTTP Request|API[FastAPI Server]
    API-->|Load Model|Model[PhoWhisper-CTC Model]
    API-->|Push Metrics|Prometheus[Prometheus]
    API-->|Send Traces|Jaeger[Jaeger]
    Prometheus-->|Visualize|Grafana[Grafana Dashboards]

    subgraph "Monitoring & Observability"
        Prometheus
        Grafana
        Jaeger
    end

    subgraph "Inference Pipeline"
        Model-->|Audio Processing|Feature[Feature Extraction]
        Feature-->|CTC Decoding|Transcription[Text Transcription]
    end

    CI[CI/CD Pipeline]-->|Build & Deploy|Deployment

    subgraph "Deployment Options"
        Docker[Docker Compose]
        K8s[Kubernetes]
    end

    Deployment-->Docker
    Deployment-->K8s

    class User,UI primary
    class API,Model accent
    class Prometheus,Grafana,Jaeger secondary
    class Docker,K8s deploy
```

## 🧠 Model Architecture

The model improves over traditional encoder-decoder ASR systems by replacing the decoder with a CTC head:

```mermaid
graph LR
    Input[Audio Input]-->Encoder[PhoWhisper Encoder]
    Encoder-->Feature[Feature Maps]
    Feature-->CTC[CTC Head]
    CTC-->Output[Text Output]

    style Encoder fill:#f9f,stroke:#333
    style CTC fill:#bbf,stroke:#333
```

### Advantages of CTC Architecture

- **Faster inference**: 2-3x faster than encoder-decoder models
- **Simpler training**: No need for autoregressive decoding
- **Reduced complexity**: Fewer parameters, smaller memory footprint
- **Streaming-friendly**: Better for real-time applications

## 🛠️ ONNX Optimization

The model supports ONNX export with INT8 quantization for faster inference on CPU:

### Benefits

- **INT8 Quantization**: 3-4x speedup over FP32 with minimal accuracy loss
- **Memory Reduction**: ~75% smaller memory footprint
- **CPU Optimization**: Better cache utilization and vector operations

### Using ONNX Models

```python
from src.models.inference_model import create_asr_model

# Create model with ONNX optimization
model = create_asr_model(
    model_id="tuandunghcmut/PhoWhisper-tiny-CTC",
    device="cpu",
    model_type="onnx"  # Use ONNX optimized version
)

# Transcribe audio
result = model.transcribe("path/to/audio.wav")
print(result["text"])
```

## 🚢 Deployment Options

### Option 1: Docker Compose (Quick Start)

```bash
# Clone the repository
git clone https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc.git
cd Convert-PhoWhisper-ASR-from-encdec-to-ctc

# Run the full stack (API, UI, and monitoring)
docker-compose -f docker/docker-compose.base.yml \
               -f docker/docker-compose.api.yml \
               -f docker/docker-compose.ui.yml \
               -f docker/docker-compose.monitoring.yml up -d
```

This will start:
- FastAPI Server: http://localhost:8000
- Streamlit UI: http://localhost:8501
- Monitoring Stack:
  - Prometheus: http://localhost:9090
  - Grafana: http://localhost:3000 (username: admin, password: admin)
  - Jaeger: http://localhost:16686

### Option 2: Kubernetes on Digital Ocean

For production deployment with high availability and scalability:

```bash
# Navigate to the k8s directory
cd k8s

# Make scripts executable (on Linux/Mac)
chmod +x setup.sh monitoring-setup.sh

# Run the setup script
./setup.sh
```

The Kubernetes deployment provides:
- API Service with 3 replicas and anti-affinity
- Load-balanced UI service
- Comprehensive monitoring with Helm charts
- Automated setup and deployment scripts

#### Kubernetes Architecture

```mermaid
graph TD
    subgraph "Digital Ocean Kubernetes"
        subgraph "ASR System Namespace"
            API[API Deployment<br>3 replicas]
            UI[UI Deployment]
            APIService[API Service<br>LoadBalancer]
            UIService[UI Service<br>LoadBalancer]
        end

        subgraph "Monitoring Namespace"
            Prometheus[Prometheus]
            Grafana[Grafana]
            AlertManager[Alert Manager]
        end

        subgraph "Observability Namespace"
            Jaeger[Jaeger All-in-One]
        end
    end

    User[User]-->|Access|APIService
    User-->|Access|UIService
    User-->|View Dashboards|Grafana
    User-->|View Traces|Jaeger

    API-->Prometheus
    API-->Jaeger

    class API,UI accent
    class Prometheus,Grafana,Jaeger secondary
```

## 🖥️ API Usage

### Endpoints

- `GET /`: API information
- `GET /models`: List available models
- `GET /languages`: List supported languages
- `GET /health`: Health check
- `POST /transcribe`: Transcribe audio file

Example transcription request:

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.wav" \
  -F "model=phowhisper-tiny-ctc" \
  -F "language=vi"
```

## 📊 Monitoring and Metrics

### Available Metrics

- **API Metrics**:
  - HTTP request count by endpoint and status
  - Request duration histograms
  - Endpoint errors and exceptions

- **ASR Model Metrics**:
  - Transcription count by model and language
  - Transcription processing time
  - Audio duration statistics
  - Inference operations in progress

- **System Metrics**:
  - Container CPU and memory usage
  - Host metrics via Node Exporter
  - Network traffic and disk I/O

### Grafana Dashboards

The system comes with pre-configured Grafana dashboards:
- ASR System Overview
- API Performance
- Node and Container metrics

## 🔄 CI/CD Pipeline

The project includes a complete CI/CD pipeline using Jenkins:

```mermaid
graph TD
    Code[Code Changes]-->GH[GitHub Repository]
    GH-->|Webhook|Jenkins[Jenkins Server]
    Jenkins-->|Build|Images[Docker Images]
    Images-->|Push|Registry[Docker Registry]
    Jenkins-->|Deploy|Docker[Docker Compose]
    Jenkins-->|Deploy|K8s[Kubernetes]

    style Jenkins fill:#f9f,stroke:#333
    style Registry fill:#bbf,stroke:#333
```

### Pipeline Features

- Automated builds on code changes
- Docker image creation and versioning
- Deployment to Docker Compose or Kubernetes
- Monitoring setup automation

## 🔍 Model Performance

- **Word Error Rate (WER)**: ~30% on the VietBud500 test set
- **Real-time factor**: <0.5x (more than 2x faster than real-time)
- **Memory usage**: <400MB
- **Processing time**: ~0.02 seconds per example on standard GPU hardware

## 🛠️ Development Guide

### Setting Up Local Development Environment

```bash
# Clone the repository
git clone https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc.git
cd Convert-PhoWhisper-ASR-from-encdec-to-ctc

# Install API dependencies
pip install -r api/requirements.txt

# Install UI dependencies
pip install -r ui/requirements.txt

# Run API server
cd api
uvicorn app:app --host 0.0.0.0 --port 8000

# In another terminal, run the UI
cd ui
streamlit run app.py
```

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

## 🔧 Troubleshooting

### Common Issues

1. **API server fails to start**:
   - Check if the port is already in use
   - Verify network connectivity

2. **Model fails to load**:
   - Check available memory
   - Verify model paths and credentials

3. **Poor transcription quality**:
   - Check audio quality (sampling rate, noise levels)
   - Try different model sizes (tiny, base, small)

4. **Kubernetes deployment issues**:
   - Verify Digital Ocean API token permissions
   - Check resource constraints and quotas
   - Examine pod logs for detailed errors

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- VinAI Research for the PhoWhisper-Tiny model
- The creators of the VietBud500 dataset
- The FastAPI, PyTorch, and Streamlit communities
