import io
import os
from typing import Any, Dict, List, Optional, Union

import plotly.graph_objects as go
import requests
import streamlit as st
from pydub import AudioSegment

# Default configurations
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_MODELS = ["phowhisper-tiny-ctc"]
DEFAULT_LANGUAGES = ["vi", "en", "auto"]
DEFAULT_MODEL_TYPES = ["pytorch", "onnx"]
DEFAULT_TIMEOUT = 10  # Increased timeout in seconds

# Flag to enable/disable trace propagation
ENABLE_TRACE_PROPAGATION = os.environ.get("ENABLE_TRACE_PROPAGATION", "true").lower() == "true"

# Initialize trace propagation if enabled
if ENABLE_TRACE_PROPAGATION:
    try:
        from opentelemetry import trace
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        
        # Global propagator
        propagator = TraceContextTextMapPropagator()
        
        # Function to inject trace context into headers
        def inject_trace_context(headers):
            """Inject trace context into request headers for distributed tracing"""
            if not headers:
                headers = {}
            
            # Get current span and inject trace context
            current_span = trace.get_current_span()
            if current_span and hasattr(current_span, "get_span_context"):
                # Carrier is our headers dict that will be mutated by inject
                propagator.inject(carrier=headers)
            
            return headers
    except ImportError:
        # Fall back if OpenTelemetry is not installed
        def inject_trace_context(headers):
            """Dummy function when OpenTelemetry is not available"""
            return headers or {}
else:
    # Dummy function when trace propagation is disabled
    def inject_trace_context(headers):
        """Dummy function when trace propagation is disabled"""
        return headers or {}


# API Communication Functions
def get_api_url() -> str:
    """
    Attempts to connect to the configured API_URL.
    If connection fails, tries to connect to host machine on port 8000.
    Returns the working API URL.
    """
    api_url = os.getenv("API_URL", DEFAULT_API_URL)

    # First try the configured API_URL
    try:
        # Add trace context for distributed tracing
        headers = inject_trace_context({})
        response = requests.get(f"{api_url}/health", headers=headers, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            st.success(f"✅ Connected to API at {api_url}")
            return api_url
    except requests.exceptions.RequestException:
        st.warning(f"⚠️ Cannot connect to {api_url}, trying fallback...")

    # Try fallback to localhost if API_URL is not localhost
    if api_url != "http://localhost:8000":
        try:
            # Add trace context for distributed tracing
            headers = inject_trace_context({})
            response = requests.get("http://localhost:8000/health", headers=headers, timeout=DEFAULT_TIMEOUT)
            if response.status_code == 200:
                st.success("✅ Connected to API at http://localhost:8000")
                return "http://localhost:8000"
        except requests.exceptions.RequestException:
            st.warning("⚠️ Cannot connect to localhost, trying host.docker.internal...")

    # Try fallback to host machine
    try:
        # Add trace context for distributed tracing
        headers = inject_trace_context({})
        response = requests.get("http://host.docker.internal:8000/health", headers=headers, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            st.success("✅ Connected to API at http://host.docker.internal:8000")
            return "http://host.docker.internal:8000"
    except requests.exceptions.RequestException:
        st.error("❌ Failed to connect to API. Please check if the API server is running.")

    # If all attempts fail, return the original URL but show a warning
    st.error(
        f"❌ Could not establish API connection. Using {api_url} but functionality may be limited."
    )
    return api_url


def check_api_status(api_url: str) -> bool:
    """Check if the API is available and responding"""
    try:
        # Add trace context for distributed tracing
        headers = inject_trace_context({})
        response = requests.get(f"{api_url}/", headers=headers, timeout=DEFAULT_TIMEOUT)
        return response.status_code == 200
    except:
        return False


def get_available_models(api_url: str) -> list[dict[str, Any]]:
    """Get list of available models from API"""
    try:
        # Add trace context for distributed tracing
        headers = inject_trace_context({})
        response = requests.get(f"{api_url}/models", headers=headers, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            # Handle both cases - either a dictionary with 'models' key or direct list
            if isinstance(data, dict) and "models" in data:
                return data["models"]
            elif isinstance(data, list):
                return data
            else:
                st.warning(f"Unexpected model data format: {data}")
                return DEFAULT_MODELS
        else:
            st.error(f"Failed to get models: {response.text}")
            return DEFAULT_MODELS
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return DEFAULT_MODELS


def get_supported_languages(api_url: str) -> list[str]:
    """Get list of supported languages from API"""
    try:
        # Add trace context for distributed tracing
        headers = inject_trace_context({})
        response = requests.get(f"{api_url}/languages", headers=headers, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            # Handle both cases - either a dictionary with 'languages' key or direct list
            if isinstance(data, dict) and "languages" in data:
                return data["languages"]
            elif isinstance(data, list):
                return data
            else:
                st.warning(f"Unexpected language data format: {data}")
                return DEFAULT_LANGUAGES
        else:
            st.error(f"Failed to get languages: {response.text}")
            return DEFAULT_LANGUAGES
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return DEFAULT_LANGUAGES


def get_model_types(api_url: str) -> list[str]:
    """Get list of available model types from API"""
    try:
        # Add trace context for distributed tracing
        headers = inject_trace_context({})
        response = requests.get(f"{api_url}/", headers=headers, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            # Check if MODEL_TYPES is in the response
            if "model_types" in data:
                return data["model_types"]
            else:
                return DEFAULT_MODEL_TYPES
        else:
            st.error(f"Failed to get model types: {response.text}")
            return DEFAULT_MODEL_TYPES
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return DEFAULT_MODEL_TYPES


def transcribe_audio(
    api_url: str, audio_file: Any, model: str, language: str, model_type: str | None = None
) -> dict[str, Any] | None:
    """Send audio for transcription to API and process result"""
    try:
        files = {"file": audio_file}
        data = {"model": model, "language": language}
        if model_type:
            data["model_type"] = model_type

        # Add trace context for distributed tracing
        headers = inject_trace_context({})

        with st.spinner("Transcribing audio..."):
            response = requests.post(
                f"{api_url}/transcribe", 
                files=files, 
                data=data, 
                headers=headers,
                timeout=DEFAULT_TIMEOUT * 3  # Longer timeout for transcription
            )

        if response.status_code == 200:
            result = response.json()
            # Add a success flag and default confidence for UI
            result["success"] = True
            if "confidence" not in result:
                result["confidence"] = 0.8  # Default confidence if not provided by API
            # Map 'text' to 'transcription' for UI consistency if needed
            if "text" in result and "transcription" not in result:
                result["transcription"] = result["text"]

            # Post-process the transcription to remove the first two strange characters
            if "transcription" in result and len(result["transcription"]) > 2:
                result["transcription"] = result["transcription"][2:]
                
            # Display trace ID if available
            if "trace_id" in result and result["trace_id"]:
                st.info(f"Trace ID: {result['trace_id']}")

            return result
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to transcribe: {e}")
        return None


# UI Helper Functions
def create_confidence_chart(confidence: float) -> go.Figure:
    """Create a gauge chart for displaying confidence"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={"text": "Confidence"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "red"},
                    {"range": [50, 75], "color": "orange"},
                    {"range": [75, 100], "color": "green"},
                ],
            },
        )
    )
    fig.update_layout(height=250)
    return fig


def check_audio_duration(audio_bytes: bytes) -> float | None:
    """Check if audio duration is within acceptable range, returns duration if valid"""
    try:
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        duration_seconds = len(audio) / 1000

        if duration_seconds < 0.5:
            st.warning("⚠️ Recording too short. Please record at least 0.5 seconds.")
            return None
        if duration_seconds > 60:
            st.warning("⚠️ Recording too long. Please keep it under 60 seconds.")
            return None

        return duration_seconds
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None


def format_model_option(model: str | dict[str, Any]) -> str:
    """Format model object for display in selectbox"""
    if isinstance(model, dict) and "id" in model:
        return f"{model['name']} - {model.get('description', '')}"
    return model


def get_model_id(model_display: str | dict[str, Any]) -> str:
    """Extract model ID from model display object"""
    if isinstance(model_display, dict) and "id" in model_display:
        return model_display["id"]
    return model_display
