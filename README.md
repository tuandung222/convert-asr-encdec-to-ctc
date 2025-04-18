# Vietnamese Speech Recognition with CTC

This project implements an Automatic Speech Recognition (ASR) system for Vietnamese using CTC (Connectionist Temporal Classification). The model is based on a modified version of PhoWhisper, converting its encoder-decoder architecture to a CTC-based architecture for more efficient training and inference.

## Project Structure

```
speech_processing/
├── api/               # FastAPI implementation
├── configs/           # Configuration files for model and training
├── k8s/               # Kubernetes deployment files
│   └── helm/          # Helm charts for k8s deployment
├── monitoring/        # Monitoring configuration
├── src/               # Source code
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model architecture definitions
│   ├── training/      # Training utilities
│   └── utils/         # Helper functions
├── scripts/           # Training and inference scripts
├── notebooks/         # Jupyter notebooks for exploration
├── run.py             # Main entry point
└── tests/             # Unit tests
```

## MLOps Architecture

This project follows MLOps best practices with a modern cloud-native architecture:

1. **API Service** - FastAPI-based REST API for speech recognition
2. **Model Service** - Optimized for CPU inference with caching
3. **Observability Stack**:
   - Prometheus for metrics collection
   - Grafana for dashboard visualization
   - Jaeger for distributed tracing
4. **CI/CD Pipeline** with Jenkins
5. **Container Orchestration** using Kubernetes with 3 replicas
6. **Infrastructure as Code** with Helm charts

See `architecture.md` for detailed architecture documentation.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc.git
cd Convert-PhoWhisper-ASR-from-encdec-to-ctc
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Local Model Usage

The project provides a single entry point (`run.py`) for all operations:

```bash
# Training
python run.py train

# Inference
python run.py infer --audio /path/to/audio_file_or_directory

# Interactive Demo
python run.py app
```

### Docker Deployment

Deploy the entire stack with Docker Compose:

```bash
docker-compose up -d
```

This will start:
- The ASR API service
- Prometheus for metrics
- Grafana for dashboards (accessible at http://localhost:3000)
- Jaeger for tracing (accessible at http://localhost:16686)

### Kubernetes Deployment

Deploy to Kubernetes:

```bash
# Deploy API service
kubectl apply -f k8s/api-deployment.yaml

# Deploy monitoring using Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm dependency update k8s/helm/monitoring
helm install asr-monitoring k8s/helm/monitoring
```

## Model Architecture

The model uses the encoder from PhoWhisper and replaces the decoder with a CTC head. This modification:
- Simplifies the architecture
- Enables faster training and inference
- Maintains competitive accuracy

Key components:
- PhoWhisper encoder for feature extraction
- Custom CTC head for Vietnamese ASR
- Efficient data loading with PyTorch Lightning

## Training

1. Configure your training parameters in `configs/training_config.yaml`
2. Run training:
```bash
python run.py train
```

Training features:
- Mixed precision training
- Multi-GPU support
- Wandb logging
- Checkpointing

## CPU Inference

This model is optimized for CPU inference, making it suitable for deployment in environments without GPUs.

### Option 1: Direct Script Inference

For batch processing of audio files:

```bash
python run.py infer --audio /path/to/audio_file_or_directory --device cpu
```

Parameters:
- `--audio`: Path to an audio file or directory containing audio files
- `--output`: Path to save transcription results (default: outputs/transcriptions.txt)
- `--checkpoint`: Path to model checkpoint (optional)
- `--device`: Device to run inference on (default: cpu)

### Option 2: Interactive Demo with Gradio

For interactive testing with a web interface:

```bash
python run.py app --device cpu
```

Parameters:
- `--checkpoint`: Path to model checkpoint (optional)
- `--device`: Device to run inference on (default: cpu)
- `--share`: Share the app publicly through Gradio
- `--port`: Port to run the app on (default: 7860)

### Option 3: REST API

Access the model through a REST API:

```bash
# Start the API service
cd api
uvicorn main:app --host 0.0.0.0 --port 8000

# Make a transcription request
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio/file.wav"
```

### Download Pre-trained Model

You can download the pre-trained model from HuggingFace:

```python
from huggingface_hub import hf_hub_download

# Download the checkpoint
checkpoint_path = hf_hub_download(
    repo_id="tuandunghcmut/PhoWhisper-tiny-CTC",
    filename="best-val_wer=0.3986.ckpt",
    local_dir="./checkpoints",
)
```

Then use it for inference:

```bash
python run.py infer --audio /path/to/audio --checkpoint ./checkpoints/best-val_wer=0.3986.ckpt --device cpu
```

## CPU Performance

On a standard CPU, the model achieves:
- Real-time factor (RTF): ~0.3-0.5x (2-3x faster than real-time)
- Memory usage: < 500MB
- Minimal latency for short audio clips

## Monitoring & Observability

### Metrics

Access Prometheus metrics at:
- http://localhost:9090 (when running with Docker Compose)
- https://your-domain/prometheus (when deployed to Kubernetes)

### Dashboards

Access Grafana dashboards at:
- http://localhost:3000 (when running with Docker Compose)
- https://your-domain/grafana (when deployed to Kubernetes)

Default credentials: admin/admin

### Tracing

Access Jaeger tracing UI at:
- http://localhost:16686 (when running with Docker Compose)
- https://your-domain/jaeger (when deployed to Kubernetes)

## Evaluation

The model achieves competitive results on Vietnamese speech recognition:
- WER (Word Error Rate): 41% on the VietBud500 test set
- CER (Character Error Rate): Comparable to state-of-the-art models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{vietnamese_asr_ctc,
  author = {Dung Vo Pham Tuan},
  title = {Vietnamese Speech Recognition with CTC},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc}
}
```

## Acknowledgments

- VINAI for the PhoWhisper model
- VietBud500 dataset creators 