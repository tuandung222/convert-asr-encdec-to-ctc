#!/bin/bash

# Create the directory structure
mkdir -p src/models src/utils

# Create empty __init__.py files
touch src/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py

# Create inference_model.py
cat > src/models/inference_model.py << 'EOF'
import logging

logger = logging.getLogger(__name__)

class ASRInferenceModel:
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device
        logger.info(f"Initialized ASR model from {model_path} on {device}")
    
    def transcribe(self, audio_path):
        logger.info(f"Transcribing audio: {audio_path}")
        return {"text": "This is a placeholder transcription.", "duration": 1.0}

def create_asr_model(model_path, device="cpu", model_type="pytorch"):
    logger.info(f"Creating ASR model: {model_path}, type: {model_type}, device: {device}")
    return ASRInferenceModel(model_path, device)
EOF

# Create metrics.py
cat > api/metrics.py << 'EOF'
from prometheus_client import Counter, Gauge, Histogram
import time

REQUESTS = Counter("http_requests_total", "Total HTTP Requests", ["method", "endpoint", "status"])
REQUEST_DURATION = Histogram("http_request_duration_seconds", "HTTP Request Duration", ["method", "endpoint"])
TRANSCRIPTIONS = Counter("asr_transcriptions_total", "Total Transcriptions", ["model", "language", "status"])
TRANSCRIPTION_DURATION = Histogram("asr_transcription_duration_seconds", "Transcription Duration", ["model", "language"])
AUDIO_DURATION = Histogram("asr_audio_duration_seconds", "Audio Duration", ["format"])
INFERENCE_IN_PROGRESS = Gauge("asr_inference_in_progress", "Number of inferences in progress", ["model"])
MODEL_LOADING_TIME = Histogram("asr_model_loading_time_seconds", "Model Loading Time", ["model", "checkpoint", "type"])
MODEL_LOAD_FAILURES = Counter("asr_model_load_failures_total", "Failed Model Loads", ["model", "type"])

class Timer:
    def __init__(self, histogram, labels=None):
        self.histogram = histogram
        self.labels = labels or {}
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.histogram.labels(**self.labels).observe(time.time() - self.start_time)
EOF

echo "Created all required stub files" 