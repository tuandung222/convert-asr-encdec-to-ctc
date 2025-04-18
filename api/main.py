#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Import prometheus metrics
from prometheus_fastapi_instrumentator import Instrumentator

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

# Initialize model
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing model...")
    
    # Set up configuration for inference
    config = setup_config_for_inference(
        model_config_path=os.getenv("MODEL_CONFIG_PATH", "configs/model_config.yaml"),
        inference_config_path=os.getenv("INFERENCE_CONFIG_PATH", "configs/inference_config.yaml"),
        override_values={
            "inference.device": os.getenv("INFERENCE_DEVICE", "cpu"),
            "model.checkpoint_path": os.getenv("MODEL_CHECKPOINT_PATH", "checkpoints/best-val_wer.ckpt"),
            "model.huggingface_repo_id": os.getenv("HUGGINGFACE_REPO_ID", "tuandunghcmut/PhoWhisper-tiny-CTC"),
            "model.huggingface_filename": os.getenv("HUGGINGFACE_FILENAME", "best-val_wer=0.3986.ckpt"),
        }
    )
    
    # Create model manager
    app.state.model_manager = ModelManager(config)
    logger.info("Model initialized successfully!")

# Define response models
class TranscriptionResponse(BaseModel):
    text: str
    confidence: Optional[float] = None
    processing_time: float
    audio_duration: float

class BatchTranscriptionResponse(BaseModel):
    results: List[TranscriptionResponse]
    total_processing_time: float
    total_audio_duration: float

# Define API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not hasattr(app.state, "model_manager"):
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Model not initialized"}
        )
    return {"status": "ok", "message": "Service is healthy"}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Transcribe audio file to text.
    
    - **file**: Audio file to transcribe (WAV, MP3, FLAC, OGG, M4A)
    
    Returns the transcribed text, confidence score, and processing information.
    """
    with tracer.start_as_current_span("transcribe_audio") as span:
        try:
            # Validate file
            if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                raise HTTPException(status_code=400, detail="Unsupported file format")
            
            # Save uploaded file to temp directory
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file.write(await file.read())
                temp_file_path = temp_file.name
            
            span.set_attribute("audio.filename", file.filename)
            span.set_attribute("audio.path", temp_file_path)
            
            # Process with model
            import time
            start_time = time.time()
            
            with tracer.start_as_current_span("model_inference") as model_span:
                # Get audio duration
                from src.utils.audio import get_audio_duration
                audio_duration = get_audio_duration(temp_file_path)
                
                # Transcribe audio
                transcription = app.state.model_manager.transcribe_file(temp_file_path)
                
                processing_time = time.time() - start_time
                model_span.set_attribute("model.inference_time", processing_time)
                model_span.set_attribute("audio.duration", audio_duration)
            
            # Clean up temp file (in background to not block response)
            if background_tasks:
                background_tasks.add_task(os.unlink, temp_file_path)
            
            response = TranscriptionResponse(
                text=transcription,
                processing_time=processing_time,
                audio_duration=audio_duration,
                confidence=None  # We don't have confidence scores yet
            )
            
            span.set_attribute("processing.time", processing_time)
            span.set_attribute("processing.rtf", processing_time / audio_duration if audio_duration > 0 else 0)
            
            return response
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            span.record_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-transcribe", response_model=BatchTranscriptionResponse)
async def batch_transcribe(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
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
                if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                    continue
                
                # Save uploaded file to temp directory
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
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
                        confidence=None
                    )
                )
            
            # Clean up temp files (in background to not block response)
            if background_tasks:
                for temp_file_path in temp_files:
                    background_tasks.add_task(os.unlink, temp_file_path)
            
            total_processing_time = time.time() - total_start_time
            
            span.set_attribute("batch.total_processing_time", total_processing_time)
            span.set_attribute("batch.total_audio_duration", total_audio_duration)
            span.set_attribute("batch.average_rtf", 
                             total_processing_time / total_audio_duration if total_audio_duration > 0 else 0)
            
            return BatchTranscriptionResponse(
                results=results,
                total_processing_time=total_processing_time,
                total_audio_duration=total_audio_duration
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
        reload=os.getenv("ENVIRONMENT", "development") == "development"
    ) 