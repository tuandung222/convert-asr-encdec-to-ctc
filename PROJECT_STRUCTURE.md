# Vietnamese ASR Project Structure

This document provides a detailed explanation of the project structure for the Vietnamese Automatic Speech Recognition (ASR) system. The project implements a CTC-based model derived from PhoWhisper for Vietnamese speech recognition.

## Overview

The project follows a modular architecture with clear separation of concerns:

```
vietnamese-asr/
├── api/                  # FastAPI implementation
├── src/                  # Core source code
│   ├── models/           # Model definitions
│   ├── utils/            # Utility functions
│   └── app/              # App implementations
├── ui/                   # UI implementation
├── examples/             # Example audio files
├── notebooks/            # Development notebooks
├── scripts/              # Utility scripts
└── ...                   # Configuration and setup files
```

## Core Components

### 1. ASR Model Implementation (`src/models/`)

The core of the project is the ASR model implementation in `src/models/inference_model.py`. This file contains several key classes:

- **`PhoWhisperCTCModel`**: The CTC-based model that consists of:
  - A WhisperEncoder for feature extraction
  - A linear CTC head for sequence prediction

- **`ASRInferenceModel`**: A wrapper class that handles:
  - Model loading and initialization
  - Inference pipeline for audio transcription
  - Pre/post-processing of audio data

- **`ONNXASRInferenceModel`**: An extension that provides ONNX support for optimized inference on CPU:
  - Automatic conversion from PyTorch to ONNX format
  - INT8 quantization for 3-4x faster CPU inference
  - Optimized CTC decoding using NumPy vectorization
  - Compatible with the base model API

The model structure simplifies the original PhoWhisper architecture by replacing the encoder-decoder approach with a CTC-based approach. This significantly improves inference speed while maintaining reasonable accuracy.

### 2. ONNX Optimization (`src/models/inference_model.py`)

The project includes an optimized ONNX implementation for faster inference:

- **Model Conversion**: Automatic conversion from PyTorch to ONNX
- **INT8 Quantization**: Dynamic quantization for better CPU performance
- **CTC Decoding**: Vectorized implementation for faster post-processing
- **Fallback Mechanism**: Graceful fallback to PyTorch if ONNX is unavailable

The ONNX implementation provides several benefits:
- 3-4x faster inference on CPU
- Reduced memory footprint (~75% reduction)
- Better CPU utilization and lower power consumption
- Same accuracy as the PyTorch implementation

### 3. API Server (`api/`)

The API server is implemented using FastAPI and provides a RESTful interface for the ASR model:

- **`api/app.py`**: The main API implementation with:
  - Endpoints for transcription
  - Model management with caching
  - Middleware for request/response handling
  - Health check and informational endpoints

The API follows best practices with proper error handling, request validation, and response formatting.

### 4. User Interfaces

The project provides two user interfaces:

#### Streamlit UI (`ui/`)

- **`ui/app.py`**: A Streamlit application that provides:
  - Audio file upload and recording
  - Transcription display
  - History tracking
  - Visualization of results

#### Gradio Demo (`app.py` and `src/app/gradio_demo.py`)

- **`app.py`**: A Gradio interface for simple demonstration with:
  - Audio recording and file upload
  - Simple transcription display
  - Example audio files

These interfaces make the ASR model accessible to both technical and non-technical users.

### 5. Entry Points and Utilities

- **`app.py`**: A Gradio interface for simple demonstration with:
  - Audio recording and file upload
  - Simple transcription display
  - Example audio files

- **`test_model.py`**: A test script for model validation with:
  - Model loading
  - Inference testing
  - Performance metrics collection

- **`scripts/app.py`**: Utility script for running the application components

### 6. Training and Evaluation (`notebooks/`)

- **`notebooks/training.py`**: Implementation of the training pipeline:
  - Data loading with `VietBud500DataModule`
  - Model definition with `PhoWhisperLightningModule`
  - Training loop with PyTorch Lightning
  - Evaluation callbacks

- **`notebooks/evaluation_after_training.py`**: Evaluation script for trained models:
  - Checkpoint loading
  - Test set evaluation
  - Performance metrics calculation
  - Error analysis

### 7. Docker and Deployment

- **`Dockerfile`**: Main Dockerfile for the application
- **`docker-compose.yml`**: Composition of all services (API, UIs, Monitoring)
- **`api/Dockerfile`**: Specific Dockerfile for the API service
- **`src/app/Dockerfile.gradio`**: Specific Dockerfile for the Gradio demo

## Detailed File Description

### Source Files (`src/`)

- **`src/models/inference_model.py`**:
  - Core ASR model implementation
  - Model loading and checkpoint management
  - Audio transcription pipeline
  - CTC decoding functionality

- **`src/utils/__init__.py`**:
  - Utility functions package initialization
  - Common helper functions

### API Files (`api/`)

- **`api/app.py`**:
  - FastAPI server implementation
  - REST API endpoints
  - Middleware for request processing
  - Error handling and validation

- **`api/__init__.py`**:
  - API package initialization

### UI Files (`ui/`)

- **`ui/app.py`**:
  - Streamlit application
  - Audio recording and file upload interface
  - History tracking and visualization
  - API communication

- **`ui/requirements.txt`**:
  - UI-specific dependencies

### Scripts and Entry Points

- **`app.py`**:
  - Gradio demo application
  - Simple interface for ASR

- **`test_model.py`**:
  - Model testing script
  - Performance validation

### Configuration Files

- **`requirements.txt`**:
  - Project dependencies

- **`docker-compose.yml`**:
  - Services composition
  - Environment configuration
  - Network and volume setup

## Data Flow

1. **Audio Input**:
   - User uploads or records audio through UI (Streamlit/Gradio)
   - Audio is sent to the API server

2. **API Processing**:
   - API receives the audio file
   - Passes it to the ASR model
   - Returns transcription results

3. **ASR Model Pipeline**:
   - Audio is loaded and preprocessed
   - Features are extracted using WhisperFeatureExtractor
   - Encoder processes features
   - CTC head generates logits
   - CTC decoding produces the final transcription

4. **Response Handling**:
   - Transcription is returned to the UI
   - Results are displayed to the user
   - History is updated (in Streamlit UI)

## Technical Details

### Model Architecture

The ASR model uses a CTC-based approach:

1. **Encoder**: WhisperEncoder from PhoWhisper
   - Processes audio features
   - Extracts contextual representations

2. **CTC Head**: Linear layer
   - Maps encoder outputs to vocabulary size
   - Produces per-frame token probabilities

3. **Decoding**:
   - Argmax decoding for inference
   - Removal of blank tokens and duplicates
   - Mapping to text with tokenizer

### Training Approach

The model is trained using:

1. **Data**: VietBud500 dataset (Vietnamese speech)
2. **Framework**: PyTorch Lightning
3. **Optimizer**: AdamW with weight decay
4. **Learning Rate**: Cosine scheduling with warmup
5. **Loss**: CTC Loss with pad token as blank
6. **Evaluation**: Word Error Rate (WER)

### Deployment Architecture

The deployment uses Docker with:

1. **API Container**: FastAPI server
2. **UI Containers**: Streamlit and Gradio
3. **Monitoring**: Prometheus, Grafana, Jaeger
4. **Networking**: Internal docker network

## Conclusion

The Vietnamese ASR project follows a well-structured, modular architecture that separates concerns and promotes maintainability. The combination of a fast, CTC-based model with user-friendly interfaces and robust API design creates a versatile speech recognition system that's efficient, accessible, and production-ready.
