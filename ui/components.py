import io
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from utils import check_audio_duration, transcribe_audio


# Recording components
def show_recording_steps(current_step: int) -> None:
    """Display the recording workflow steps with appropriate icons"""
    steps = ["Record your voice 🎙️", "Review the recording 👂", "Transcribe to text ✍️"]

    for i, step in enumerate(steps, 1):
        if i < current_step:
            st.markdown(f"✅ {i}. {step}")
        elif i == current_step:
            st.markdown(f"🔵 {i}. {step}")
        else:
            st.markdown(f"⚪ {i}. {step}")


def audio_recorder_component() -> bytes | None:
    """Display and handle the audio recorder component"""
    audio_bytes = audio_recorder(
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x",
    )

    if audio_bytes:
        if check_audio_duration(audio_bytes) is not None:
            return audio_bytes
    return None


def recording_instructions() -> None:
    """Display recording instructions"""
    st.markdown("### Instructions")
    st.markdown(
        """
        1. Click the microphone button to start recording
        2. Speak clearly in Vietnamese
        3. Click again to stop recording
        4. Review your recording
        5. Click "Transcribe" to process

        **Tips:**
        - Keep recordings under 60 seconds
        - Speak clearly and at a normal pace
        - Minimize background noise
        """
    )


# Transcription result components
def display_transcription_result(result: dict[str, Any], audio_bytes: bytes | None = None) -> None:
    """Display transcription results in a formatted way"""
    st.success("✓ Transcription complete")

    with st.container():
        st.markdown("### Transcription Result")

        # Display the transcription with highlighting
        st.markdown(
            f"""<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;
            border-left: 5px solid #4c8bff; margin-bottom: 20px; font-size: 18px;">
            {result['transcription']}
            </div>""",
            unsafe_allow_html=True,
        )

        # Display metrics without nested columns
        st.markdown("### Metrics")
        metric_html = f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 18px; font-weight: bold;">⏱️ Duration</div>
                <div>{result.get('duration', 0):.2f}s</div>
            </div>
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 18px; font-weight: bold;">⚡ Processing Time</div>
                <div>{result['processing_time']:.2f}s</div>
            </div>
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 18px; font-weight: bold;">🚀 Real-time Factor</div>
                <div>{result.get('real_time_factor', 0):.2f}x</div>
            </div>
        </div>
        """
        st.markdown(metric_html, unsafe_allow_html=True)

        # Additional metadata
        with st.expander("Show Details"):
            st.markdown(f"**Model:** {result['model']}")
            st.markdown(f"**Language:** {result['language']}")
            if "timestamp" in result:
                st.markdown(f"**Timestamp:** {result['timestamp']}")
            if "model_type" in result:
                st.markdown(f"**Model Type:** {result['model_type']}")
            if "trace_id" in result and result["trace_id"]:
                jaeger_url = os.environ.get("JAEGER_URL", "http://localhost:16686")
                trace_id = result["trace_id"]
                st.markdown(f"**Trace ID:** `{trace_id}`")
                st.markdown(f"[View this trace in Jaeger]({jaeger_url}/trace/{trace_id})")

        # Add download buttons - FIX: Use a horizontal layout without columns
        if audio_bytes:
            st.markdown("### Download")
            # Use HTML for a horizontal button layout instead of columns
            st.markdown(
                """
                <div style="display: flex; gap: 10px;">
                    <div style="flex: 1;"></div>
                    <div style="flex: 1;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Place buttons directly without nested columns
            st.download_button(
                label="📄 Download Transcription",
                data=result["transcription"],
                file_name=f"transcription_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt",
                mime="text/plain",
            )
            st.download_button(
                label="🔊 Download Audio",
                data=audio_bytes,
                file_name=f"recording_{time.strftime('%Y-%m-%d_%H-%M-%S')}.wav",
                mime="audio/wav",
            )
        else:
            st.download_button(
                label="📄 Download Transcription",
                data=result["transcription"],
                file_name=f"transcription_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt",
                mime="text/plain",
            )


def process_recording(
    audio_bytes: bytes, selected_model: str, selected_language: str, api_url: str
) -> dict[str, Any] | None:
    """Process recording and return transcription result"""
    # Create a temporary file for the audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name

    try:
        # Open the temp file for sending to API
        with open(tmp_file_path, "rb") as audio_file:
            result = transcribe_audio(api_url, audio_file, selected_model, selected_language)

        if result and result.get("success", False):
            # Add source information
            result["source"] = "Recording"
            result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            return result
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


# History components
def display_history(history: list[dict[str, Any]]) -> bool:
    """Display transcription history as interactive table"""
    if not history:
        st.info("No transcriptions yet. Upload or record audio to get started.")
        return False

    # Create a dataframe from the history
    history_df = pd.DataFrame(history)

    # Format the dataframe for display
    display_df = history_df.copy()
    if "timestamp" in display_df.columns:
        display_df = display_df.sort_values("timestamp", ascending=False)

    # Add source icons
    if "source" in display_df.columns:
        display_df["source"] = display_df["source"].apply(
            lambda x: "🎙️ Recording" if x == "Recording" else "📁 Upload"
        )

    # Select columns to display
    display_columns = [
        "timestamp",
        "source",
        "model",
        "language",
        "processing_time",
        "transcription",
    ]
    display_columns = [col for col in display_columns if col in display_df.columns]

    # Display the history as a dataframe with an expanded view option
    st.dataframe(display_df[display_columns], use_container_width=True)

    # Allow viewing detailed history
    if st.button("View Detailed History"):
        st.json(history)

    # Add export and clear history buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Export as CSV"):
            csv = display_df[display_columns].to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"transcription_history_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv",
                mime="text/csv",
            )
    with col2:
        if st.button("🗑️ Clear History"):
            return True
    return False


# System status components
def display_api_status(api_url: str, api_connected: bool) -> None:
    """Display API status information"""
    st.subheader("API Status")
    if api_connected:
        try:
            import requests

            response = requests.get(f"{api_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                st.success("✅ API is healthy")

                # Display health data in a more structured format
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Status", "Online")
                    if "uptime" in health_data:
                        st.metric("Uptime", f"{health_data['uptime']:.2f}s")
                with col2:
                    if "version" in health_data:
                        st.metric("Version", health_data["version"])
                    if "models_loaded" in health_data:
                        st.metric("Models Loaded", health_data["models_loaded"])

                # Show tracing status if available
                if "tracing_enabled" in health_data:
                    if health_data["tracing_enabled"]:
                        st.success("✅ Distributed Tracing is enabled")
                    else:
                        st.warning("⚠️ Distributed Tracing is disabled")

                # Show full health data in expandable section
                with st.expander("Show detailed health information"):
                    st.json(health_data)
            else:
                st.error(f"❌ API health check failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot connect to API: {e}")
    else:
        st.error("❌ API is not connected")
        st.info(f"Make sure the API is running at {api_url}")


def display_tracing_info(jaeger_url: str) -> None:
    """Display Jaeger tracing information"""
    st.subheader("Distributed Tracing")

    # Check if Jaeger URL is defined
    if not jaeger_url:
        jaeger_url = os.environ.get("JAEGER_URL", "http://localhost:16686")

    # Add information about tracing
    st.markdown(
        """
    ### Distributed Tracing with Jaeger

    This application uses OpenTelemetry to capture distributed traces across the UI and API components.
    Traces are exported to Jaeger, which provides a visualization interface for analyzing request flows.

    **Key benefits:**
    - End-to-end visibility of request processing
    - Identification of performance bottlenecks
    - Debugging of errors across service boundaries
    - Analysis of system behavior in real-time
    """
    )

    # Add button to open Jaeger UI
    if st.button("🔍 Open Jaeger UI"):
        st.markdown(f"[Open Jaeger UI in new tab]({jaeger_url})")
        st.components.iframe(jaeger_url, height=300)

    # Add a section about how to use trace IDs
    with st.expander("How to use trace IDs"):
        st.markdown(
            """
        When you perform a transcription, a unique trace ID is generated and displayed in the result.

        To find a specific trace in Jaeger:
        1. Open the Jaeger UI
        2. Select the 'asr-api' service from the dropdown
        3. Click 'Find Traces'
        4. Enter the trace ID in the 'Tags' field using format: `trace_id=<your_trace_id>`
        5. Click 'Find Traces' to locate your specific trace

        You can also explore traces by time range, service, operation, and duration.
        """
        )


def display_model_info(api_url: str) -> None:
    """Display model information from API"""
    st.subheader("Available Models")
    try:
        import requests

        response = requests.get(f"{api_url}/models")
        if response.status_code == 200:
            models_data = response.json()

            # Create a cleaner display of models
            for model in models_data:
                with st.container():
                    st.markdown(
                        f"""<div style="background-color: #f0f2f6; padding: 10px;
                        border-radius: 5px; margin-bottom: 10px;">
                        <strong>{model['id']}</strong>: {model.get('description', 'No description')}
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    if "features" in model:
                        features = ", ".join(model["features"])
                        st.markdown(f"**Features:** {features}")
        else:
            st.error(f"❌ Could not retrieve model information")
    except Exception as e:
        st.error(f"❌ Error getting model information: {e}")


# Styling components
def local_css() -> None:
    """Add custom CSS to improve the UI"""
    st.markdown(
        """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton button {
        border-radius: 20px;
        padding: 2px 15px;
        font-weight: 500;
    }
    .stDownloadButton button {
        border-radius: 20px;
        padding: 2px 15px;
        font-weight: 500;
    }
    /* Custom header with gradient */
    .header-container {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #3B82F6;
    }
    /* Custom tab styling */
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #1E3A8A;
    }
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.8rem;
        color: #6B7280;
    }
    /* Button container styling */
    .download-buttons-container {
        display: flex;
        gap: 10px;
    }
    .download-button {
        flex: 1;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def header() -> None:
    """Display a custom header with logo and title"""
    st.markdown(
        """
        <div class="header-container">
            <h1 class="header-title">🎙️ Vietnamese Automatic Speech Recognition</h1>
            <p class="header-subtitle">Powered by PhoWhisper-CTC | Fast and accurate transcription</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer() -> None:
    """Display a custom footer"""
    st.markdown(
        """
        <div class="footer">
            <hr>
            <p>Vietnamese ASR powered by PhoWhisper-CTC | Built with Streamlit | Metrics by Prometheus & Grafana</p>
            <p>© 2024 | <a href="https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc" target="_blank">GitHub Repository</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
