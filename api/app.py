import os
import sys
import time
import uuid
import tempfile
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response
import torch

# Add Prometheus metrics
from prometheus_client import make_asgi_app
import prometheus_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model
try:
    from src.models.inference_model import create_asr_model, ASRInferenceModel
    from src.metrics import (
        REQUESTS, REQUEST_DURATION, TRANSCRIPTIONS, TRANSCRIPTION_DURATION,
        AUDIO_DURATION, MODEL_LOADING_TIME, INFERENCE_IN_PROGRESS, Timer
    )
except ImportError:
    logger.error("Failed to import required modules. Make sure the project structure is correct.")
    raise

# Define supported models
MODELS = {
    "phowhisper-tiny-ctc": "tuandunghcmut/PhoWhisper-tiny-CTC",
}

# Define supported languages
LANGUAGES = ["vi"]

# Default model and language
DEFAULT_MODEL = "phowhisper-tiny-ctc"
DEFAULT_LANGUAGE = "vi"

# API models
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    time: str
    device: str


class TranscriptionResponse(BaseModel):
    id: str
    text: str
    duration: float
    processing_time: float
    real_time_factor: float
    language: str
    model: str
    timestamp: str


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str


# Initialize FastAPI app
app = FastAPI(
    title="PhoWhisper CTC ASR API",
    description="API for Vietnamese Automatic Speech Recognition using a CTC-based model",
    version="1.0.0",
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a dictionary to store the loaded models
model_cache = {}

def get_model(model_name: str = DEFAULT_MODEL) -> ASRInferenceModel:
    """Get or load a model from the cache"""
    global model_cache
    
    # Set device based on environment or availability
    device = os.environ.get("INFERENCE_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    # Get the HuggingFace model path
    model_path = MODELS.get(model_name)
    if not model_path:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not found")
    
    # Create a cache key that includes the model name and device
    cache_key = f"{model_name}_{device}"
    
    # Check if the model is already loaded
    if cache_key not in model_cache:
        logger.info(f"Loading model {model_name} on {device}")
        try:
            # Track model loading time
            with Timer(MODEL_LOADING_TIME, {"model": model_name, "checkpoint": model_path}):
                model_cache[cache_key] = create_asr_model(model_path, device)
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    return model_cache[cache_key]


# Add request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Add request ID to response headers
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# Add metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    method = request.method
    path = request.url.path
    
    # Skip metrics endpoint
    if path == "/metrics":
        return await call_next(request)
    
    start_time = time.time()
    
    # Track request
    with Timer(REQUEST_DURATION, {"method": method, "endpoint": path}):
        response = await call_next(request)
    
    # Record request count with status
    status = response.status_code
    REQUESTS.labels(method=method, endpoint=path, status=status).inc()
    
    # Add response time header
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


@app.get("/", tags=["Info"])
async def root():
    """Get API information"""
    return {
        "name": "PhoWhisper CTC ASR API",
        "version": "1.0.0",
        "description": "API for Vietnamese Automatic Speech Recognition using a CTC-based model",
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "GET /models": "List available models",
            "GET /languages": "List supported languages",
            "POST /transcribe": "Transcribe audio file",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check(request: Request):
    """Health check endpoint"""
    # Check if the default model is loaded
    model_loaded = False
    device = "unknown"
    
    try:
        model = get_model()
        model_loaded = True
        device = model.device
    except Exception:
        model_loaded = False
    
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "time": datetime.now().isoformat(),
        "device": device,
    }


@app.get("/models", response_model=List[ModelInfo], tags=["Info"])
async def list_models():
    """List available ASR models"""
    models = []
    for model_id, path in MODELS.items():
        models.append({
            "id": model_id,
            "name": path.split("/")[-1],
            "description": f"Vietnamese ASR model based on {path}"
        })
    return models


@app.get("/languages", response_model=List[str], tags=["Info"])
async def list_languages():
    """List supported languages"""
    return LANGUAGES


@app.post("/transcribe", response_model=TranscriptionResponse, tags=["Transcription"])
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    language: str = Form(DEFAULT_LANGUAGE),
):
    """Transcribe an audio file"""
    # Validate language
    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Language {language} not supported")
    
    # Validate model
    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model {model} not available")
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file_path = temp_file.name
        content = await file.read()
        temp_file.write(content)
    
    try:
        # Get the model
        asr_model = get_model(model)
        
        # Track transcription start
        TRANSCRIPTIONS.labels(model=model, language=language, status="started").inc()
        
        # Track inference in progress
        INFERENCE_IN_PROGRESS.labels(model=model).inc()
        
        # Process the audio
        start_time = time.time()
        
        # Use the timer for transcription duration
        with Timer(TRANSCRIPTION_DURATION, {"model": model, "language": language}):
            result = asr_model.transcribe(temp_file_path)
        
        processing_time = time.time() - start_time
        
        # Track audio duration
        audio_format = os.path.splitext(file.filename)[1].lstrip('.')
        AUDIO_DURATION.labels(format=audio_format).observe(result.get("duration", 0.0))
        
        # Track transcription success
        TRANSCRIPTIONS.labels(model=model, language=language, status="success").inc()
        
        # Create the response
        response = {
            "id": request.state.request_id,
            "text": result.get("text", ""),
            "duration": result.get("duration", 0.0),
            "processing_time": processing_time,
            "real_time_factor": processing_time / (result.get("duration", 1.0)),
            "language": language,
            "model": model,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Log the transcription
        logger.info(f"Transcription completed: {len(result.get('text', ''))} chars in {processing_time:.2f}s")
        
        return response
    except Exception as e:
        # Track transcription failure
        TRANSCRIPTIONS.labels(model=model, language=language, status="failure").inc()
        
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")
    finally:
        # Decrement in-progress count
        INFERENCE_IN_PROGRESS.labels(model=model).dec()
        
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# Run the app if executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port) 