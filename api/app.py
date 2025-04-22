import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Add Prometheus metrics
import prometheus_client
import psutil
import torch
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# OpenTelemetry imports for distributed tracing
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from prometheus_client import Gauge, multiprocess
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set up multiprocess metrics collection for Prometheus
prometheus_multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
if prometheus_multiproc_dir:
    logger.info(f"Using prometheus multiprocess directory: {prometheus_multiproc_dir}")
    if not os.path.exists(prometheus_multiproc_dir):
        os.makedirs(prometheus_multiproc_dir, exist_ok=True)
    prometheus_client.multiprocess.start_http_server_in_different_process = lambda **kwargs: None

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model
try:
    from api.metrics import MODEL_LOAD_FAILURES  # Add new metric for tracking model load failures
    from api.metrics import (
        AUDIO_DURATION,
        INFERENCE_IN_PROGRESS,
        MODEL_LOADING_TIME,
        REQUEST_DURATION,
        REQUESTS,
        TRANSCRIPTION_DURATION,
        TRANSCRIPTIONS,
        Timer,
    )
    from src.models.inference_model import ASRInferenceModel, create_asr_model

    # from src.models.improved_inference_model import ASRInferenceModel, create_asr_model
except ImportError:
    logger.error("Failed to import required modules. Make sure the project structure is correct.")
    raise

# Global tracer provider
tracer_provider = None


# Setup OpenTelemetry tracing
def setup_tracing():
    """Configure OpenTelemetry with Jaeger exporter"""
    global tracer_provider

    # Check if tracing is enabled
    tracing_enabled = os.environ.get("ENABLE_TRACING", "true").lower() == "true"
    if not tracing_enabled:
        logger.info("Distributed tracing is disabled")
        return None

    # Configure Jaeger endpoint
    jaeger_host = os.environ.get("JAEGER_HOST", "jaeger")
    jaeger_port = int(os.environ.get("JAEGER_PORT", "6831"))

    # Set up tracer provider with service name
    service_name = os.environ.get("SERVICE_NAME", "asr-api")
    resource = Resource.create({"service.name": service_name})

    # Create and set the tracer provider
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    # Create Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=jaeger_host,
        agent_port=jaeger_port,
    )

    # Add span processor to the tracer provider
    tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

    logger.info(f"Configured Jaeger tracing - endpoint: {jaeger_host}:{jaeger_port}")
    return tracer_provider


# Tracer for our app
def get_tracer():
    """Get tracer for the API service"""
    if tracer_provider:
        return trace.get_tracer("asr-api")
    return None


# Add system metrics for CPU and memory in a way that avoids duplicate registration
# Use function to create metrics on demand rather than at module level
def create_process_metrics():
    """Create process metrics with better error handling and debug logging"""
    try:
        # Create metrics with unique registry to avoid duplicate registration errors
        process_cpu_percent = Gauge(
            "process_cpu_percent",
            "CPU utilization percentage of the API process",
            ["pid", "process_name"],  # Added process_name for better identification
            registry=None,  # Don't auto-register
        )

        process_memory_usage = Gauge(
            "process_memory_usage_bytes",
            "Memory usage of the API process in bytes",
            ["pid", "process_name", "type"],  # Added process_name for better identification
            registry=None,  # Don't auto-register
        )

        process_open_fds = Gauge(
            "process_open_file_descriptors",
            "Number of open file descriptors",
            ["pid", "process_name"],  # Added process_name for better identification
            registry=None,  # Don't auto-register
        )

        # Register metrics if they don't exist yet
        try:
            prometheus_client.REGISTRY.register(process_cpu_percent)
            prometheus_client.REGISTRY.register(process_memory_usage)
            prometheus_client.REGISTRY.register(process_open_fds)
            logger.info("✅ Process metrics registered successfully")
        except ValueError as e:
            logger.info(f"Process metrics already registered, using existing metrics: {e}")
            # metrics will still be available in registry

        return process_cpu_percent, process_memory_usage, process_open_fds
    except Exception as e:
        logger.error(f"❌ Failed to create process metrics: {e}")

        # Return dummy metrics that won't fail when used
        class DummyMetric:
            def labels(self, **kwargs):
                return self

            def set(self, value):
                pass

        return DummyMetric(), DummyMetric(), DummyMetric()


# Background metrics collection
def collect_process_metrics():
    """Collect process metrics in the background with better error handling"""
    try:
        # Get process info
        pid = os.getpid()
        pid_str = str(pid)
        process = psutil.Process(pid)
        process_name = "asr-api"

        # Create metrics only once
        process_cpu_percent, process_memory_usage, process_open_fds = create_process_metrics()
        logger.info(f"🔄 Starting metrics collection for PID {pid_str}")

        # Set initial values to ensure metrics exist
        process_cpu_percent.labels(pid=pid_str, process_name=process_name).set(0)
        process_memory_usage.labels(pid=pid_str, process_name=process_name, type="rss").set(0)
        process_memory_usage.labels(pid=pid_str, process_name=process_name, type="vms").set(0)
        process_open_fds.labels(pid=pid_str, process_name=process_name).set(0)

        # Log metric initialization
        logger.info(f"✅ Metrics initialized for PID {pid_str}")

        collect_count = 0
        while True:
            try:
                # CPU percentage (interval=None means since last call)
                # First call will return 0, so we do a quick call first
                if collect_count == 0:
                    process.cpu_percent(interval=None)  # First call to initialize
                    time.sleep(0.1)

                cpu_percent = process.cpu_percent(interval=None)
                process_cpu_percent.labels(pid=pid_str, process_name=process_name).set(cpu_percent)

                # Memory usage
                memory_info = process.memory_info()
                process_memory_usage.labels(pid=pid_str, process_name=process_name, type="rss").set(
                    memory_info.rss
                )
                process_memory_usage.labels(pid=pid_str, process_name=process_name, type="vms").set(
                    memory_info.vms
                )

                # Open file descriptors
                open_fds = process.num_fds()
                process_open_fds.labels(pid=pid_str, process_name=process_name).set(open_fds)

                # Periodically log metrics for debugging
                collect_count += 1
                if collect_count % 12 == 0:  # Log every minute (at 5s intervals)
                    logger.info(
                        f"📊 Current metrics - CPU: {cpu_percent:.1f}%, "
                        f"Memory (RSS): {memory_info.rss/1024/1024:.1f}MB, "
                        f"FDs: {open_fds}"
                    )
            except Exception as e:
                logger.error(f"❌ Error collecting process metrics: {e}")

            # Sleep for 5 seconds before next collection
            time.sleep(5)
    except Exception as e:
        logger.error(f"❌ Fatal error in metrics collection thread: {e}")


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
    tracing_enabled: bool = False


class TranscriptionResponse(BaseModel):
    id: str
    text: str
    duration: float
    processing_time: float
    real_time_factor: float
    language: str
    model: str
    timestamp: str
    trace_id: str | None = None


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
@app.get("/metrics")
async def metrics():
    if prometheus_multiproc_dir:
        registry = prometheus_client.CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        content = prometheus_client.generate_latest(registry)
    else:
        content = prometheus_client.generate_latest(prometheus_client.REGISTRY)
    return Response(content=content, media_type="text/plain")


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

# Preload models configuration
PRELOAD_MODELS = os.environ.get("PRELOAD_MODELS", "true").lower() == "true"
# MODEL_TYPES = os.environ.get("MODEL_TYPES", "pytorch").split(",")  # pytorch,onnx
# MODEL_TYPES = os.environ.get("MODEL_TYPES", "onnx").split(",")  # pytorch,onnx
MODEL_TYPES = ["pytorch", "onnx"]


def get_model(model_name: str = DEFAULT_MODEL, model_type: str = None) -> ASRInferenceModel:
    """Get or load a model from the cache"""
    global model_cache

    # Get tracer for span creation
    tracer = get_tracer()

    # Create span context if tracing is enabled
    if tracer:
        with tracer.start_as_current_span("get_model") as span:
            span.set_attribute("model.name", model_name)
            span.set_attribute("model.type", model_type or MODEL_TYPES[0])
            return _get_model_internal(model_name, model_type)
    else:
        return _get_model_internal(model_name, model_type)


def _get_model_internal(
    model_name: str = DEFAULT_MODEL, model_type: str = None
) -> ASRInferenceModel:
    """Internal function to get or load a model from the cache"""
    global model_cache

    # Set device based on environment or availability
    device = os.environ.get("INFERENCE_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # Get the HuggingFace model path
    model_path = MODELS.get(model_name)
    if not model_path:
        # Record error in span if tracing enabled
        current_span = trace.get_current_span()
        if hasattr(current_span, "is_recording") and current_span.is_recording():
            current_span.set_status(StatusCode.ERROR)
            current_span.record_exception(Exception(f"Model {model_name} not found"))

        raise HTTPException(status_code=400, detail=f"Model {model_name} not found")

    # Use specified model_type or default from environment
    if model_type is None:
        model_type = MODEL_TYPES[0]  # Use first type as default

    # Create a cache key that includes the model name, type and device
    cache_key = f"{model_name}_{model_type}_{device}"

    # Check if the model is already loaded
    if cache_key not in model_cache:
        logger.info(f"Loading model {model_name} (type: {model_type}) on {device}")
        try:
            # Track model loading time
            with Timer(
                MODEL_LOADING_TIME,
                {"model": model_name, "checkpoint": model_path, "type": model_type},
            ):
                # Add span information if tracing enabled
                current_span = trace.get_current_span()
                if hasattr(current_span, "is_recording") and current_span.is_recording():
                    current_span.set_attribute("model.loading", True)
                    current_span.set_attribute("model.path", model_path)
                    current_span.set_attribute("model.device", device)

                model_cache[cache_key] = create_asr_model(model_path, device, model_type=model_type)

                # Record successful model loading in span
                if hasattr(current_span, "is_recording") and current_span.is_recording():
                    current_span.set_attribute("model.loaded", True)

            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            # Record error in span if tracing enabled
            current_span = trace.get_current_span()
            if hasattr(current_span, "is_recording") and current_span.is_recording():
                current_span.set_status(StatusCode.ERROR)
                current_span.record_exception(e)

            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    return model_cache[cache_key]


@app.on_event("startup")
async def startup_event():
    """Preload models when application starts"""
    # Setup tracing
    if setup_tracing():
        # Check if OpenTelemetry middleware should be skipped
        skip_middleware = os.environ.get("SKIP_OTEL_MIDDLEWARE", "false").lower() == "true"
        if not skip_middleware:
            try:
                # Instrument FastAPI with OpenTelemetry
                FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics")
                logger.info("✅ OpenTelemetry instrumentation enabled for FastAPI")
            except RuntimeError as e:
                logger.warning(f"⚠️ Failed to add OpenTelemetry middleware: {e}")
                logger.info("Continuing without OpenTelemetry middleware")
        else:
            logger.info("⏩ Skipping OpenTelemetry middleware as requested by environment variable")

    # Start background metrics collection
    metrics_thread = threading.Thread(target=collect_process_metrics, daemon=True)
    metrics_thread.start()
    logger.info("Started background metrics collection")

    if PRELOAD_MODELS:
        logger.info("Preloading models on startup...")
        total_models = len(MODELS) * len(MODEL_TYPES)
        successful_loads = 0
        failed_loads = 0

        for model_name in MODELS:
            for model_type in MODEL_TYPES:
                try:
                    logger.info(
                        f"Preloading model {model_name} ({model_type})... [{successful_loads + failed_loads + 1}/{total_models}]"
                    )
                    start_time = time.time()
                    model = get_model(model_name, model_type)

                    # Perform a warmup inference to ensure the model is fully initialized
                    try:
                        logger.info(f"Running warmup inference for {model_name} ({model_type})...")
                        # Create a small silence audio file in memory for warmup
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            temp_file_path = temp_file.name
                            # Write a tiny 0.1s silence WAV file (empty but valid audio)
                            import numpy as np
                            from scipy.io import wavfile

                            sample_rate = 16000  # Standard sample rate
                            duration = 0.1  # Short duration for quick warmup
                            data = np.zeros(int(sample_rate * duration), dtype=np.int16)
                            wavfile.write(temp_file_path, sample_rate, data)

                        # Run inference on the silent audio
                        _ = model.transcribe(temp_file_path)

                        # Clean up the temporary file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)

                        logger.info(f"Warmup inference completed for {model_name} ({model_type})")
                    except Exception as e:
                        logger.warning(
                            f"Warmup inference failed for {model_name} ({model_type}): {str(e)}"
                        )
                        # Continue even if warmup fails - model is still loaded

                    load_time = time.time() - start_time
                    logger.info(
                        f"✓ Successfully preloaded model {model_name} ({model_type}) in {load_time:.2f}s"
                    )
                    successful_loads += 1
                except Exception as e:
                    logger.error(f"✗ Error preloading model {model_name} ({model_type}): {str(e)}")
                    # Track failure in metrics if the metric exists
                    if "MODEL_LOAD_FAILURES" in globals():
                        MODEL_LOAD_FAILURES.labels(model=model_name, type=model_type).inc()
                    failed_loads += 1
                    # Continue with other models even if one fails

        logger.info(
            f"Model preloading complete: {successful_loads} succeeded, {failed_loads} failed"
        )
    else:
        logger.info("Model preloading disabled")


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
        "tracing_enabled": tracer_provider is not None,
    }


@app.get("/models", response_model=list[ModelInfo], tags=["Info"])
async def list_models():
    """List available ASR models"""
    models = []
    for model_id, path in MODELS.items():
        models.append(
            {
                "id": model_id,
                "name": path.split("/")[-1],
                "description": f"Vietnamese ASR model based on {path}",
            }
        )
    return models


@app.get("/languages", response_model=list[str], tags=["Info"])
async def list_languages():
    """List supported languages"""
    return LANGUAGES


@app.post("/transcribe", response_model=TranscriptionResponse, tags=["Transcription"])
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    language: str = Form(DEFAULT_LANGUAGE),
    model_type: str = Form(None),
):
    """Transcribe an audio file"""
    # Get tracer for span creation
    tracer = get_tracer()
    current_span = trace.get_current_span()
    trace_id = None

    # Extract trace context if present to link with other systems
    if hasattr(current_span, "get_span_context") and hasattr(
        current_span.get_span_context(), "trace_id"
    ):
        trace_id = format(current_span.get_span_context().trace_id, "032x")

    # Set span attributes if tracing is enabled
    if hasattr(current_span, "is_recording") and current_span.is_recording():
        current_span.set_attribute("request.id", request.state.request_id)
        current_span.set_attribute("audio.filename", file.filename)
        current_span.set_attribute("audio.model", model)
        current_span.set_attribute("audio.language", language)
        if model_type:
            current_span.set_attribute("audio.model_type", model_type)

    # Validate language
    if language not in LANGUAGES:
        if hasattr(current_span, "is_recording") and current_span.is_recording():
            current_span.set_status(StatusCode.ERROR)
            current_span.record_exception(Exception(f"Language {language} not supported"))

        raise HTTPException(status_code=400, detail=f"Language {language} not supported")

    # Validate model
    if model not in MODELS:
        if hasattr(current_span, "is_recording") and current_span.is_recording():
            current_span.set_status(StatusCode.ERROR)
            current_span.record_exception(Exception(f"Model {model} not available"))

        raise HTTPException(status_code=400, detail=f"Model {model} not available")

    # Validate model type if provided
    if model_type is not None and model_type not in MODEL_TYPES:
        if hasattr(current_span, "is_recording") and current_span.is_recording():
            current_span.set_status(StatusCode.ERROR)
            current_span.record_exception(Exception(f"Model type {model_type} not supported"))

        raise HTTPException(
            status_code=400,
            detail=f"Model type {model_type} not supported. Available types: {', '.join(MODEL_TYPES)}",
        )

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as temp_file:
        temp_file_path = temp_file.name
        content = await file.read()
        temp_file.write(content)

        # Record file size in span if tracing enabled
        if hasattr(current_span, "is_recording") and current_span.is_recording():
            current_span.set_attribute("audio.size_bytes", len(content))

    try:
        # Create nested span for model loading if tracing enabled
        if tracer:
            with tracer.start_as_current_span("load_model") as model_span:
                model_span.set_attribute("model.name", model)
                model_span.set_attribute("model.type", model_type or MODEL_TYPES[0])
                # Get the model
                asr_model = get_model(model, model_type)
        else:
            # Get the model without tracing
            asr_model = get_model(model, model_type)

        # Track transcription start
        TRANSCRIPTIONS.labels(model=model, language=language, status="started").inc()

        # Track inference in progress
        INFERENCE_IN_PROGRESS.labels(model=model).inc()

        # Process the audio
        start_time = time.time()

        # Create nested span for transcription if tracing enabled
        if tracer:
            with tracer.start_as_current_span("transcribe") as transcribe_span:
                transcribe_span.set_attribute("audio.file", file.filename)
                transcribe_span.set_attribute("audio.language", language)

                # Use the timer for transcription duration
                with Timer(TRANSCRIPTION_DURATION, {"model": model, "language": language}):
                    result = asr_model.transcribe(temp_file_path)

                # Add result attributes to span
                transcribe_span.set_attribute("transcription.length", len(result.get("text", "")))
        else:
            # Use the timer for transcription duration without tracing
            with Timer(TRANSCRIPTION_DURATION, {"model": model, "language": language}):
                result = asr_model.transcribe(temp_file_path)

        processing_time = time.time() - start_time

        # Get duration from result or default to 1.0 to avoid division by zero
        audio_duration = max(1.0, result.get("duration", 1.0))

        # Record audio duration in span if tracing enabled
        if hasattr(current_span, "is_recording") and current_span.is_recording():
            current_span.set_attribute("audio.duration_seconds", audio_duration)
            current_span.set_attribute("processing.duration_seconds", processing_time)

        # Track audio duration if available
        audio_format = os.path.splitext(file.filename)[1].lstrip(".")
        AUDIO_DURATION.labels(format=audio_format).observe(audio_duration)

        # Track transcription success
        TRANSCRIPTIONS.labels(model=model, language=language, status="success").inc()

        # Calculate real-time factor safely
        real_time_factor = processing_time / max(1.0, audio_duration)

        # Record real-time factor in span if tracing enabled
        if hasattr(current_span, "is_recording") and current_span.is_recording():
            current_span.set_attribute("processing.real_time_factor", real_time_factor)

        # Create the response
        response = {
            "id": request.state.request_id,
            "text": result.get("text", ""),
            "duration": audio_duration,
            "processing_time": processing_time,
            "real_time_factor": real_time_factor,
            "language": language,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "trace_id": trace_id,
        }

        # Log the transcription
        logger.info(
            f"Transcription completed: {len(result.get('text', ''))} chars in {processing_time:.2f}s"
        )

        return response
    except Exception as e:
        # Track transcription failure
        TRANSCRIPTIONS.labels(model=model, language=language, status="failure").inc()

        # Record error in span if tracing enabled
        if hasattr(current_span, "is_recording") and current_span.is_recording():
            current_span.set_status(StatusCode.ERROR)
            current_span.record_exception(e)

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
    uvicorn.run("api.app:app", host="0.0.0.0", port=port)
