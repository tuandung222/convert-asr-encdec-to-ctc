# Vietnamese Speech Recognition with PhoWhisper-CTC

This project implements an Automatic Speech Recognition (ASR) system for Vietnamese using a CTC-based model derived from PhoWhisper. The model replaces the encoder-decoder architecture with a more efficient CTC-based approach for faster inference and simplified training.

## Features

- Fast and accurate Vietnamese speech recognition
- CTC-based architecture for efficient inference
- Multiple UI options (Streamlit and Gradio)
- FastAPI backend for production use
- Docker support for easy deployment
- Monitoring and observability built-in

## Model Architecture

The model is based on the PhoWhisper architecture, with the following modifications:
- Retains the encoder from PhoWhisper for feature extraction
- Replaces the decoder with a CTC head for sequence modeling
- Simplifies the inference process by removing the need for autoregressive decoding

The CTC (Connectionist Temporal Classification) approach offers several advantages:
- Faster inference (2-3x faster than real-time)
- Simpler training with parallel decoding
- Reduced model complexity and memory usage
- Smaller deployment footprint

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
│   ├── grafana/          # Grafana dashboards and provisioning
│   │   └── provisioning/ # Auto-provisioning configurations
│   │       ├── dashboards/    # Dashboard configurations
│   │       │   └── json/      # Dashboard JSON definitions
│   │       └── datasources/   # Data source configurations
│   └── prometheus/       # Prometheus configuration
│       └── prometheus.yml     # Prometheus config file
├── examples/             # Example audio files
├── notebooks/            # Development notebooks
│   ├── training.py       # Training code
│   └── evaluation_after_training.py  # Evaluation code
├── scripts/              # Utility scripts
│   └── app.py            # App runner
├── app.py                # Gradio app entry point
├── run.py                # Unified entry point
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

The easiest way to run the entire stack is with Docker Compose:

```bash
docker-compose up -d
```

This will start:
- FastAPI Server (http://localhost:8000)
- Streamlit UI (http://localhost:8501)
- Gradio Demo (http://localhost:7860)
- Monitoring Stack
  - Prometheus (http://localhost:9090)
  - Grafana (http://localhost:3000)
  - Jaeger (http://localhost:16686)

### Option 2: Run Components Individually

#### API Server

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
python -m api.app
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

The UI will be available at http://localhost:8501

#### Gradio Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Gradio demo
python run.py app --device cpu
```

The Gradio demo will be available at http://localhost:7860

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

- Word Error Rate (WER): ~41% on the test set
- Real-time factor: <0.5x (more than 2x faster than real-time)
- Memory usage: <500MB

## Troubleshooting

- If the API server fails to start, check if the port is already in use
- If the model fails to load, check if you have enough memory
- If the transcription is poor, try using a different model or check the audio quality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VinAI Research for the PhoWhisper-Tiny model
- The creators of the VietBud500 dataset 