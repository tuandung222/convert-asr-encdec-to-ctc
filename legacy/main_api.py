#!/usr/bin/env python

import io
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pydub import AudioSegment

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Import prometheus metrics
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.models.inference_model import create_asr_model

# Import our model manager
from src.models.model_manager import ModelManager
from src.utils.config import setup_config_for_inference
from src.utils.logging import setup_logging

# Configure logging
logger = logging.getLogger(__name__)
setup_logging()

# Configure OpenTelemetry
resource = Resource(attributes={SERVICE_NAME: "vietnamese-asr-api"})
trace_provider = TracerProvider(resource=resource)
jaeger_exporter = JaegerExporter(
    agent_host_name=os.getenv("JAEGER_HOST", "localhost"),
    agent_port=int(os.getenv("JAEGER_PORT", "6831")),
)
trace_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
trace.set_tracer_provider(trace_provider)
tracer = trace.get_tracer(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vietnamese ASR API",
    description="API for Vietnamese Speech Recognition using PhoWhisper-CTC",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Supported models (currently only one model is supported)
SUPPORTED_MODELS = ["phowhisper-tiny-ctc"]
SUPPORTED_LANGUAGES = ["vi", "en", "auto"]

# Global model cache
models = {}


def get_model(model_id: str, device: str = "cpu"):
    """Get or load the model for the given ID"""
    model_key = f"{model_id}_{device}"

    if model_key not in models:
        logger.info(f"Loading model {model_id} on {device}")

        # Map friendly model name to actual model parameters
        if model_id == "phowhisper-tiny-ctc":
            models[model_key] = create_asr_model(
                model_type="pytorch",
                model_name="vinai/PhoWhisper-tiny",
                repo_id="tuandunghcmut/PhoWhisper-tiny-CTC",
                checkpoint_filename="best-val_wer=0.3986.ckpt",
                use_cuda=(device == "cuda"),
            )
        else:
            raise ValueError(f"Unsupported model ID: {model_id}")

    return models[model_key]


# Initialize model
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the Vietnamese ASR API...")

    # Pre-load the default model
    try:
        # Use environment variable to determine device, defaulting to CPU
        device = os.environ.get("INFERENCE_DEVICE", "cpu")
        get_model("phowhisper-tiny-ctc", device)
        logger.info("Default model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading default model: {str(e)}")


# Define response models
class TranscriptionResponse(BaseModel):
    text: str
    confidence: float | None = None
    processing_time: float
    audio_duration: float


class BatchTranscriptionResponse(BaseModel):
    results: list[TranscriptionResponse]
    total_processing_time: float
    total_audio_duration: float


# Define API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Vietnamese ASR API is running",
        "supported_models": SUPPORTED_MODELS,
        "supported_languages": SUPPORTED_LANGUAGES,
    }


@app.get("/models")
async def get_models():
    """Get available models"""
    return {"models": SUPPORTED_MODELS}


@app.get("/languages")
async def get_languages():
    """Get supported languages"""
    return {"languages": SUPPORTED_LANGUAGES}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


def _process_audio_file(file_path: str, model_id: str, language: str):
    """Process audio file with the specified model"""
    try:
        # Get the model
        device = os.environ.get("INFERENCE_DEVICE", "cpu")
        model = get_model(model_id, device)

        # Transcribe the audio
        result = model.transcribe(file_path)

        # Add additional information to the result
        result["model"] = model_id
        if language != "auto":  # If a specific language was requested
            result["language"] = language

        return result
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return {"success": False, "error": str(e)}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("phowhisper-tiny-ctc"),
    language: str = Form("vi"),
):
    """
    Transcribe an audio file.

    Args:
        file: Audio file to transcribe
        model: Model ID to use for transcription
        language: Language of the audio

    Returns:
        Transcription result
    """
    # Validate model
    if model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400, detail=f"Model not supported. Choose from {SUPPORTED_MODELS}"
        )

    # Validate language
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400, detail=f"Language not supported. Choose from {SUPPORTED_LANGUAGES}"
        )

    try:
        # Check file type
        if not file.content_type.startswith("audio/"):
            logger.warning(f"Received file with content type: {file.content_type}")

        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file.filename.split('.')[-1]}"
        ) as temp_file:
            # Read the uploaded file and write to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Process the audio file
            result = _process_audio_file(temp_file_path, model, language)

            # Add filename to result
            result["filename"] = file.filename

            return result
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.post("/batch-transcribe", response_model=BatchTranscriptionResponse)
async def batch_transcribe(
    files: list[UploadFile] = File(...), background_tasks: BackgroundTasks = None
):
    """
    Transcribe multiple audio files in a batch.

    - **files**: List of audio files to transcribe

    Returns transcriptions for all files along with processing information.
    """
    with tracer.start_as_current_span("batch_transcribe") as span:
        span.set_attribute("batch.size", len(files))

        try:
            results = []
            temp_files = []
            total_start_time = time.time()
            total_audio_duration = 0

            # Process each file
            for file in files:
                # Validate file
                if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                    continue

                # Save uploaded file to temp directory
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(file.filename)[1]
                ) as temp_file:
                    temp_file.write(await file.read())
                    temp_file_path = temp_file.name
                    temp_files.append(temp_file_path)

                # Process with model
                with tracer.start_as_current_span(f"transcribe_{file.filename}") as file_span:
                    start_time = time.time()

                    # Get audio duration
                    audio_duration = get_audio_duration(temp_file_path)
                    total_audio_duration += audio_duration

                    # Transcribe audio
                    transcription = app.state.model_manager.transcribe_file(temp_file_path)

                    processing_time = time.time() - start_time

                    file_span.set_attribute("audio.filename", file.filename)
                    file_span.set_attribute("model.inference_time", processing_time)
                    file_span.set_attribute("audio.duration", audio_duration)

                # Add result
                results.append(
                    TranscriptionResponse(
                        text=transcription,
                        processing_time=processing_time,
                        audio_duration=audio_duration,
                        confidence=None,
                    )
                )

            # Clean up temp files (in background to not block response)
            if background_tasks:
                for temp_file_path in temp_files:
                    background_tasks.add_task(os.unlink, temp_file_path)

            total_processing_time = time.time() - total_start_time

            span.set_attribute("batch.total_processing_time", total_processing_time)
            span.set_attribute("batch.total_audio_duration", total_audio_duration)
            span.set_attribute(
                "batch.average_rtf",
                total_processing_time / total_audio_duration if total_audio_duration > 0 else 0,
            )

            return BatchTranscriptionResponse(
                results=results,
                total_processing_time=total_processing_time,
                total_audio_duration=total_audio_duration,
            )

        except Exception as e:
            logger.error(f"Error in batch transcription: {e}")
            span.record_exception(e)
            raise HTTPException(status_code=500, detail=str(e))


# Add OpenTelemetry instrumentation to FastAPI
FastAPIInstrumentor.instrument_app(app)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENVIRONMENT", "development") == "development",
    )
