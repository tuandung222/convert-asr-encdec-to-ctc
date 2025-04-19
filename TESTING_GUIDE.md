# Vietnamese ASR System Testing Guide

This guide provides a comprehensive approach for testing the Vietnamese ASR (Automatic Speech Recognition) system's core components locally. We'll test the FastAPI server and Streamlit demo directly on your Linux machine while using Docker only for the monitoring services.

## Overview

The testing process involves:

1. Setting up the environment
2. Running monitoring services via Docker Compose
3. Running the FastAPI server locally
4. Running the Streamlit app locally
5. Testing key MVP features
6. Verifying integration with monitoring

## 1. Environment Setup

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Git
- Required Python packages

### Clone Repository

If you haven't already done so:

```bash
git clone https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc.git
cd Convert-PhoWhisper-ASR-from-encdec-to-ctc
```

### Create Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
source venv/bin/activate

# Install main requirements
pip install -r requirements.txt

# Install UI-specific requirements
pip install -r ui/requirements.txt
```

### Download Model Checkpoint

Ensure the model checkpoint is available:

```bash
# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Download the checkpoint (if not already present)
wget -O checkpoints/best-val_wer=0.3986.ckpt https://huggingface.co/tuandunghcmut/PhoWhisper-tiny-CTC/resolve/main/best-val_wer%3D0.3986.ckpt
```

## 2. Running Monitoring Stack via Docker

We'll use Docker Compose to run only the monitoring services:

```bash
# Create a specific docker-compose file for monitoring only
cat > docker-compose.monitoring.yml << 'EOF'
version: '3.8'

services:
  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:v2.46.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.path=/prometheus
      - --storage.tsdb.retention.time=15d
      - --web.console.libraries=/usr/share/prometheus/console_libraries
      - --web.console.templates=/usr/share/prometheus/consoles
      - --web.enable-lifecycle
      - --web.enable-admin-api
      - --web.external-url=http://localhost:9090
      - --alertmanager.url=http://alertmanager:9093
    restart: unless-stopped
    networks:
      - asr-network
    user: "nobody:nobody"

  # AlertManager for alert management
  alertmanager:
    image: prom/alertmanager:v0.26.0
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager:/etc/alertmanager
    command:
      - --config.file=/etc/alertmanager/alertmanager.yml
      - --storage.path=/alertmanager
      - --web.external-url=http://localhost:9093
    restart: unless-stopped
    networks:
      - asr-network
    user: "nobody:nobody"

  # Grafana for dashboards
  grafana:
    image: grafana/grafana:10.2.3
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=F7aJw3kQ9pL5xYzR
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SERVER_ROOT_URL=http://localhost:3000
      - GF_SERVER_DOMAIN=localhost
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-piechart-panel
      - GF_AUTH_ANONYMOUS_ENABLED=false
      - GF_FEATURE_TOGGLES_ENABLE=publicDashboards
      - GF_UNIFIED_ALERTING_UI_DISABLE_REPROVISION=true
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - asr-network

  # Jaeger for distributed tracing
  jaeger:
    image: jaegertracing/all-in-one:1.48
    ports:
      - "6831:6831/udp"  # accept jaeger.thrift over compact thrift protocol
      - "6832:6832/udp"  # accept jaeger.thrift over binary thrift protocol
      - "16686:16686"    # UI port
      - "14268:14268"    # HTTP Collector
    environment:
      - COLLECTOR_ZIPKIN_HTTP_PORT=9411
      - COLLECTOR_OTLP_ENABLED=true
      - JAEGER_DISABLED=false
      - SPAN_STORAGE_TYPE=memory
      - METRICS_STORAGE_TYPE=prometheus
      - PROMETHEUS_SERVER_URL=http://prometheus:9090
      - SAMPLING_STRATEGIES_FILE=/etc/jaeger/sampling.json
    volumes:
      - ./monitoring/jaeger:/etc/jaeger
    restart: unless-stopped
    networks:
      - asr-network

volumes:
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  asr-network:
    driver: bridge
EOF

# Start the monitoring services
docker-compose -f docker-compose.monitoring.yml up -d
```

## 3. Running FastAPI Server Locally

Now we'll run the FastAPI server directly on the local machine:

```bash
# Set environment variables for the FastAPI server
export PORT=8000
export INFERENCE_DEVICE=cpu
export JAEGER_HOST=localhost
export JAEGER_PORT=6831
export ENVIRONMENT=development
export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus-metrics

# Create prometheus metrics directory
mkdir -p /tmp/prometheus-metrics

# Run the FastAPI server
cd api
python app.py
```

The FastAPI server should now be running at http://localhost:8000.

## 4. Running Streamlit App Locally

Open a new terminal and run the Streamlit app:

```bash
# Activate virtual environment in the new terminal
source venv/bin/activate

# Set environment variables for the Streamlit app
export API_URL=http://localhost:8000

# Run the Streamlit app
cd ui
streamlit run app.py
```

The Streamlit app should now be accessible at http://localhost:8501.

## 5. Testing MVP Features

### 5.1 Test Feature: Audio Upload

**Steps:**
1. Navigate to http://localhost:8501 in your browser
2. Click "Upload Audio File"
3. Select an audio file (WAV or MP3 format) with Vietnamese speech
4. Verify that the file uploads successfully
5. Check that transcription appears in the UI after processing

**Expected Result:** The system should transcribe the uploaded audio and display the text.

### 5.2 Test Feature: Audio Recording

**Steps:**
1. Navigate to http://localhost:8501 in your browser
2. Click on the "Record Audio" tab
3. Click "🎙️ Start Recording" to begin recording
4. Speak a Vietnamese phrase
5. Click "⏹️ Stop Recording" to finish
6. Review your recording (playback is available)
7. Click "🎯 Transcribe" to process the audio
8. Verify that transcription appears with details like confidence, model used, etc.
9. Test other options like "Record Again" or "Download Audio"

**Expected Result:** The system should guide you through the recording process with clear state indicators, then transcribe the recorded audio and display the text along with metadata.

### 5.3 Test Feature: API Direct Access

**Steps:**
1. Use cURL to test the API directly:

```bash
# Health check
curl -X GET http://localhost:8000/health

# List available models
curl -X GET http://localhost:8000/models

# Transcribe an audio file
curl -X POST http://localhost:8000/transcribe \
  -H "accept: application/json" \
  -F "file=@/path/to/audio.wav" \
  -F "model=phowhisper-tiny-ctc" \
  -F "language=vi"
```

**Expected Result:** The API should respond with appropriate JSON responses for each endpoint.

### 5.4 Test Feature: Error Handling

**Steps:**
1. Try uploading an invalid file format (e.g., a text file)
2. Try submitting a very short audio (less than 0.5 seconds)
3. Try submitting a very large file (>10MB)
4. Try submitting an audio file with non-Vietnamese speech

**Expected Result:** The system should handle errors gracefully and provide informative error messages.

## 6. Monitoring Integration Verification

### 6.1 Verify Prometheus Metrics

**Steps:**
1. Make several API requests (uploads, recordings)
2. Access Prometheus at http://localhost:9090
3. Check metrics using queries:
   - `up{job="asr-api"}` (should be 1 for the API)
   - `http_requests_total{job="asr-api"}`
   - `transcription_requests_total{job="asr-api"}`

**Expected Result:** Should see metrics being collected for the API.

### 6.2 Verify Grafana Dashboard

**Steps:**
1. Access Grafana at http://localhost:3000 (login with admin/F7aJw3kQ9pL5xYzR)
2. Navigate to the "ASR Monitoring" folder
3. Open the "Vietnamese ASR Dashboard"
4. Check that API metrics are being displayed

**Expected Result:** Dashboard should show real-time metrics from the locally running API.

### 6.3 Verify Jaeger Tracing

**Steps:**
1. Make several API requests
2. Access Jaeger UI at http://localhost:16686
3. Select "asr-api" from the Service dropdown
4. Click "Find Traces"

**Expected Result:** Should see traces for the API requests you made.

## 7. Performance Testing

### 7.1 Response Time Test

**Steps:**
1. Use the following script to test response time:

```python
import requests
import time
import statistics

url = "http://localhost:8000/transcribe"
file_path = "path/to/test/audio.wav"
response_times = []

for i in range(10):
    start_time = time.time()
    with open(file_path, "rb") as file:
        response = requests.post(
            url,
            files={"file": file},
            data={"model": "phowhisper-tiny-ctc", "language": "vi"}
        )
    end_time = time.time()
    response_time = end_time - start_time
    response_times.append(response_time)
    print(f"Request {i+1}: {response_time:.2f} seconds")

print(f"Average response time: {statistics.mean(response_times):.2f} seconds")
print(f"Min response time: {min(response_times):.2f} seconds")
print(f"Max response time: {max(response_times):.2f} seconds")
```

**Expected Result:** Response times should be reasonable for the hardware being used.

### 7.2 Concurrent Requests Test

Create a script to test how the system handles multiple concurrent requests:

```python
import requests
import concurrent.futures
import time

def make_request(file_path, i):
    start_time = time.time()
    with open(file_path, "rb") as file:
        response = requests.post(
            "http://localhost:8000/transcribe",
            files={"file": file},
            data={"model": "phowhisper-tiny-ctc", "language": "vi"}
        )
    end_time = time.time()
    return {"id": i, "time": end_time - start_time, "status": response.status_code}

file_path = "path/to/test/audio.wav"
results = []

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(make_request, file_path, i) for i in range(10)]
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())

for result in sorted(results, key=lambda x: x["id"]):
    print(f"Request {result['id']}: {result['time']:.2f}s, Status: {result['status']}")
```

**Expected Result:** The system should handle concurrent requests without failures.

### 7.3 ONNX Performance Testing

To test the ONNX optimization with INT8 quantization, use the following script:

```python
import time
import os
from src.models.inference_model import create_asr_model

def test_model_performance(audio_path, model_types=["pytorch", "onnx"]):
    """Compare performance between different model types"""
    results = {}
    
    for model_type in model_types:
        print(f"\nTesting {model_type.upper()} model:")
        
        # Load model
        start_time = time.time()
        model = create_asr_model(
            model_id="tuandunghcmut/PhoWhisper-tiny-CTC",
            device="cpu",
            model_type=model_type
        )
        load_time = time.time() - start_time
        print(f"  Model loaded in {load_time:.2f} seconds")
        
        # Run inference (multiple times to get average)
        inference_times = []
        for i in range(5):
            start_time = time.time()
            result = model.transcribe(audio_path)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            print(f"  Run {i+1}: {inference_time:.2f} seconds")
        
        # Calculate statistics
        avg_time = sum(inference_times) / len(inference_times)
        results[model_type] = {
            "load_time": load_time,
            "avg_inference_time": avg_time,
            "text": result["text"]
        }
        
        print(f"  Average inference time: {avg_time:.2f} seconds")
        print(f"  Transcription: {result['text']}")
    
    # Compare results
    if "pytorch" in results and "onnx" in results:
        speedup = results["pytorch"]["avg_inference_time"] / results["onnx"]["avg_inference_time"]
        print(f"\nONNX speedup: {speedup:.2f}x faster than PyTorch")
        
    return results

# Test with your audio file
test_model_performance("path/to/audio.wav")
```

**Expected Results:**
- ONNX with INT8 quantization should be 3-4x faster than PyTorch on CPU
- Both models should produce similar transcription results
- First-time ONNX run will be slower due to model conversion and quantization

## 8. Cleanup

When done testing:

```bash
# Stop FastAPI server (Ctrl+C in its terminal)

# Stop Streamlit app (Ctrl+C in its terminal)

# Stop monitoring services
docker-compose -f docker-compose.monitoring.yml down

# Deactivate virtual environment
deactivate
```

## Additional Notes

- If experiencing issues with the FastAPI server, check logs for detailed error messages
- For memory usage issues, consider reducing batch sizes or model complexity
- The CTC-based model should be significantly faster than the original Whisper encoder-decoder model
- For production, consider optimizing the model with ONNX or TensorRT for faster inference 