# Vietnamese Speech Recognition with PhoWhisper-CTC

This project implements an Automatic Speech Recognition (ASR) system for Vietnamese using a CTC-based model derived from PhoWhisper. The model replaces the encoder-decoder architecture with a more efficient CTC-based approach for faster inference and simplified training.

## Features

- Fast and accurate Vietnamese speech recognition
- CTC-based architecture for efficient inference
- Multiple UI options (Streamlit and Gradio)
- FastAPI backend for production use
- Docker support for easy deployment
- Monitoring and observability built-in
- ONNX optimization with INT8 quantization for CPU
- CI/CD with Jenkins and GitHub Actions

## Model Architecture

The model is based on the PhoWhisper architecture, with the following modifications:
- Retains the encoder from PhoWhisper for feature extraction
- Replaces the decoder with a CTC head for sequence modeling
- Simplifies the inference process by removing the need for autoregressive decoding
- Supports ONNX export with INT8 quantization for optimized CPU inference

The CTC (Connectionist Temporal Classification) approach offers several advantages:
- Faster inference (2-3x faster than real-time)
- Simpler training with parallel decoding
- Reduced model complexity and memory usage
- Smaller deployment footprint

## ONNX Optimization

The model supports ONNX export with INT8 quantization for faster inference on CPU:

### Benefits
- **INT8 Quantization**: 3-4x speedup over FP32 with minimal accuracy loss
- **Memory Reduction**: ~75% smaller memory footprint
- **CPU Optimization**: Better cache utilization and vector operations
- **Deployment Friendly**: Ideal for resource-constrained environments

### Using ONNX Models
To use the ONNX-optimized model:

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

The first run will automatically convert and quantize the model to INT8 ONNX format.

## Training Process

The training process involves:

1. **Data Preparation**:
   - Using a subset of the VietBud500 dataset for Vietnamese speech
   - Processing audio data with WhisperProcessor for feature extraction
   - Preparing targets using CTC-based alignment

2. **Model Setup**:
   - Starting with a pre-trained PhoWhisper encoder
   - Adding a custom CTC head consisting of a linear layer
   - Setting up CTC loss with the pad token as the blank token

3. **Training Loop**:
   - Using PyTorch Lightning for training management
   - Implementing AdamW optimizer with weight decay
   - Applying cosine learning rate scheduling with warmup
   - Training for 64 epochs with batch size of 24
   - Using mixed precision (bfloat16) for faster training

4. **Evaluation Metrics**:
   - Monitoring Word Error Rate (WER) on validation set
   - Using custom EvalCallback to track model progress
   - Implementing custom CTC decoding for prediction

## Project Structure

```
vietnamese-asr/
├── api/                  # FastAPI implementation
│   ├── __init__.py       # Package initialization
│   ├── app.py            # API server implementation
│   └── Dockerfile        # Docker config for API
├── src/                  # Core source code
│   ├── __init__.py
│   ├── models/           # Model definitions
│   │   ├── __init__.py
│   │   └── inference_model.py  # Inference model implementation
│   ├── utils/            # Utility functions
│   │   └── __init__.py
│   ├── metrics.py        # Prometheus metrics definitions
│   └── app/              # App implementations
│       └── gradio_demo.py    # Gradio demo implementation
├── ui/                   # UI implementation
│   ├── app.py            # Streamlit app
│   ├── requirements.txt  # UI dependencies
│   └── static/           # Static assets
├── monitoring/           # Monitoring configuration
├── jenkins/              # Jenkins CI/CD configuration
│   ├── Dockerfile        # Jenkins server with Docker capabilities
│   ├── Jenkinsfile       # CI/CD pipeline definition
│   ├── plugins.txt       # Required Jenkins plugins
│   └── scripts/          # CI/CD automation scripts
├── k8s/                  # Kubernetes deployment configs
├── examples/             # Example audio files
├── notebooks/            # Development notebooks
├── scripts/              # Utility scripts
├── app.py                # Gradio app entry point
├── test_model.py         # Model testing script
├── requirements.txt      # Project dependencies
├── Dockerfile            # Main Dockerfile
├── docker-compose.yml    # Docker composition
└── README.md             # Project documentation
```

## Training and Evaluation Details

### VietBud500 DataModule

The `VietBud500DataModule` handles data loading:
- Loads a subset of the VietBud500 dataset
- Preprocesses audio with WhisperProcessor
- Creates train/validation/test splits
- Implements collate_fn for batch preparation

### PhoWhisperLightningModule

The `PhoWhisperLightningModule` implements:
- CTC-based model architecture
- Forward pass logic
- CTC loss computation
- Training, validation, and test steps
- Optimizer configuration with AdamW
- CTC decoding for prediction

### Evaluation Approach

1. **During Training**:
   - Validation WER is calculated after each epoch
   - Best model checkpoint is saved based on lowest WER
   - Sample predictions are logged for manual inspection

2. **Final Evaluation**:
   - Test set evaluation using best checkpoint
   - WER calculation on full test set
   - Error analysis and performance metrics

3. **Inference Performance**:
   - Real-time factor (RTF) measurement
   - Memory usage tracking
   - System requirements assessment

## Architecture

The project consists of several components:

1. **ASR Model**: A CTC-based Vietnamese speech recognition model using PhoWhisper's encoder
2. **FastAPI Server**: RESTful API for serving the ASR model
3. **Streamlit UI**: User-friendly interface for interacting with the API
4. **Gradio Demo**: Simple demo interface for quick testing
5. **Monitoring Stack**: Prometheus, Grafana, and Jaeger for observability
6. **CI/CD Pipeline**: Jenkins and GitHub Actions for automated builds and deployments

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- 1GB+ of RAM for model inference
- 500MB+ of disk space

### Installation

Clone the repository:

```bash
git clone https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc.git
cd Convert-PhoWhisper-ASR-from-encdec-to-ctc
```

### Option 1: Run with Docker Compose (Recommended)

The project uses a modular Docker Compose structure for flexible deployment options:

```bash
# Run the full stack (API, UI, and monitoring)
docker-compose -f docker/docker-compose.base.yml \
               -f docker/docker-compose.api.yml \
               -f docker/docker-compose.ui.yml \
               -f docker/docker-compose.monitoring.yml up -d

# Or run just the API with monitoring
docker-compose -f docker/docker-compose.base.yml \
               -f docker/docker-compose.api.yml \
               -f docker/docker-compose.monitoring.yml up -d
```

This will start:
- FastAPI Server (http://localhost:8000)
- Streamlit UI (http://localhost:8501)
- Monitoring Stack
  - Prometheus (http://localhost:9090)
  - Grafana (http://localhost:3000)
  - Jaeger (http://localhost:16686)

See the `docker/README.md` for more deployment options.

### Option 2: Run Components Individually

#### API Server

```bash
# Install dependencies
pip install -r api/requirements.txt

# Run the API server
cd api
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

#### Streamlit UI

```bash
# Install UI dependencies
pip install -r ui/requirements.txt

# Run the Streamlit UI
cd ui
streamlit run app.py
```

Or:

```bash
docker-compose -f docker/docker-compose.base.yml \
               -f docker/docker-compose.ui.yml up -d
```

The UI will be available at http://localhost:8501

#### Monitoring Only (for Local Development)

```bash
# Start just the monitoring stack
docker-compose -f docker/docker-compose.base.yml \
               -f docker/docker-compose.monitoring.yml up -d
```

## CI/CD with Jenkins

The project includes Jenkins configuration for automating Docker image builds and deployment. This is particularly useful for MLOps workflows where you want to automatically build and publish Docker images.

### Setting Up Jenkins for Docker Automation

1. **Quick Setup with Docker**:
   ```bash
   # Create a volume for Jenkins data
   docker volume create jenkins_data

   # Run Jenkins with Docker capabilities
   docker run -d --name jenkins-server \
     -p 8080:8080 -p 50000:50000 \
     -v jenkins_data:/var/jenkins_home \
     -v /var/run/docker.sock:/var/run/docker.sock \
     --restart unless-stopped \
     jenkins/jenkins:lts
   ```

2. **Install Required Plugins**:
   - Get initial admin password: `docker exec jenkins-server cat /var/jenkins_home/secrets/initialAdminPassword`
   - Complete setup wizard and install suggested plugins
   - Install Docker Pipeline, Git Integration plugins

3. **Configure Docker Hub Credentials**:
   - Add credentials with ID "docker-hub-credentials"
   - Enter your Docker Hub username and password/token

4. **Create Pipeline Job**:
   - Create a new Pipeline job
   - Configure SCM to point to your repository
   - Set Jenkinsfile path to `jenkins/Jenkinsfile`

5. **Run the Pipeline**:
   - The pipeline will:
     - Build Docker images for API and UI
     - Tag images with build number and "latest"
     - Push images to Docker Hub

For more detailed instructions, see `jenkins/README.md`.

## Monitoring and Metrics

The project includes a comprehensive monitoring stack for observability:

### Metrics Collection

- **Prometheus**: Collects metrics from all components
  - API performance metrics (request rate, latency)
  - Transcription metrics (count, duration, success rate)
  - System metrics (CPU, memory usage)
  - Model performance metrics (inference time, real-time factor)

### Dashboards

- **Grafana**: Provides visualization dashboards
  - ASR system dashboard with all key metrics
  - Real-time API performance monitoring
  - Model performance tracking
  - System resource utilization

### Metrics Available

- **API Metrics**:
  - HTTP request count by endpoint and status
  - Request duration histograms
  - Endpoint errors and exceptions

- **ASR Model Metrics**:
  - Transcription count by model and language
  - Transcription processing time
  - Audio duration statistics
  - Model loading time
  - Inference operations in progress

- **System Metrics**:
  - Container CPU and memory usage
  - Host metrics via Node Exporter
  - Network traffic and disk I/O

### Accessing Metrics

- Prometheus UI: http://localhost:9090
- Grafana Dashboards: http://localhost:3000 (username: admin, password: admin)
- Raw metrics endpoint: http://localhost:8000/metrics

### Distributed Tracing

- **Jaeger**: Provides distributed tracing for request flows
  - Trace API requests from UI through to model inference
  - Identify bottlenecks in the processing pipeline
  - Debug performance issues across components

- Jaeger UI: http://localhost:16686

## Using the Model

### API Endpoints

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

### Streamlit UI

The Streamlit UI provides a more user-friendly interface for:
- Uploading audio files for transcription
- Recording audio directly in the browser with step-by-step guidance
- Viewing transcription history
- Visualizing confidence scores
- Downloading audio recordings and transcription results

### Gradio Demo

The Gradio demo provides a simple interface for:
- Uploading audio files
- Recording audio
- Viewing transcription results

## Customization

### Environment Variables

- `API_URL`: URL of the FastAPI server (default: http://localhost:8000)
- `PORT`: Port for the API server (default: 8000)
- `INFERENCE_DEVICE`: Device to run inference on (cpu or cuda, default: cpu)
- `GRADIO_SHARE`: Whether to share the Gradio demo publicly (default: false)
- `GRAFANA_URL`: URL of the Grafana dashboard (default: http://localhost:3000)

## Model Performance

- Word Error Rate (WER): ~30% on the VietBud500 test set (measured on 1000 test examples)
- Real-time factor: <0.5x (more than 2x faster than real-time)
- Memory usage: <500MB
- Processing time: ~0.02 seconds per example on standard CPU hardware

## Troubleshooting

- If the API server fails to start, check if the port is already in use
- If the model fails to load, check if you have enough memory
- If the transcription is poor, try using a different model or check the audio quality
- Jenkins Docker issues: Ensure the Docker socket is correctly mounted and Jenkins has permissions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VinAI Research for the PhoWhisper-Tiny model
- The creators of the VietBud500 dataset

> **Note**: The project doesn't use a unified `run.py` interface as mentioned in legacy documentation. Instead, each component has its specific entry point as shown in the installation instructions.

# Development

## Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency. To set up the development environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc.git
   cd Convert-PhoWhisper-ASR-from-encdec-to-ctc
   ```

2. Install dependencies:
   ```bash
   pip install -r api/requirements.txt
   ```

3. Install pre-commit:
   ```bash
   pip install pre-commit
   ```

4. Install the git hooks:
   ```bash
   pre-commit install
   ```

5. Run pre-commit on all files:
   ```bash
   pre-commit run --all-files
   ```

The pre-commit hooks will:
- Format code with Black and isort
- Run Flake8 for code linting
- Check types with MyPy
- Lint Dockerfiles with hadolint
- Run security checks with Bandit
- Format notebooks with nbqa
- And more

These hooks run automatically on every commit to ensure code quality and consistency throughout the project.
