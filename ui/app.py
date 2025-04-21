import io
import os
import tempfile
import time as time_module

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from audio_recorder_streamlit import audio_recorder
from components import (
    audio_recorder_component,
    display_api_status,
    display_history,
    display_model_info,
    display_transcription_result,
    footer,
    header,
    local_css,
    process_recording,
    recording_instructions,
    show_recording_steps,
)
from dotenv import load_dotenv
from pydub import AudioSegment

# Import our custom modules
from utils import (
    check_api_status,
    format_model_option,
    get_api_url,
    get_available_models,
    get_model_id,
    get_model_types,
    get_supported_languages,
    transcribe_audio,
)

# Set page config first - must be the very first Streamlit command
st.set_page_config(
    page_title="Vietnamese ASR", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded"
)

# Apply custom CSS
local_css()

# Load environment variables
load_dotenv()

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")


def init_session_state():
    """Initialize session state variables"""
    if "transcription_history" not in st.session_state:
        st.session_state.transcription_history = []
    if "show_metrics" not in st.session_state:
        st.session_state.show_metrics = False
    if "recording_step" not in st.session_state:
        st.session_state.recording_step = 1
    if "current_recording" not in st.session_state:
        st.session_state.current_recording = None
    if "theme" not in st.session_state:
        st.session_state.theme = "light"


def upload_tab(api_url, selected_model, selected_language):
    """Content for the Upload Audio tab"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Upload Audio File")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["mp3", "wav", "m4a", "ogg", "flac"],
        help="Upload an audio file containing Vietnamese speech to transcribe",
    )

    if uploaded_file is not None:
        # Display audio player with waveform
        st.audio(uploaded_file, format="audio/wav")

        # Model type selection (optional)
        model_types = get_model_types(api_url)
        selected_model_type = st.selectbox(
            "Select Model Type (Optional)",
            ["Automatic"] + model_types,
            help="Select the model type for transcription. 'Automatic' will let the system choose.",
        )
        model_type = None if selected_model_type == "Automatic" else selected_model_type

        # Transcribe button
        if st.button("🎯 Transcribe Audio", type="primary"):
            # Reset file position
            uploaded_file.seek(0)

            # Process the transcription
            result = transcribe_audio(
                api_url, uploaded_file, selected_model, selected_language, model_type
            )

            if result and result.get("success", False):
                # Display the results
                display_transcription_result(result)

                # Add to history with timestamp
                result["timestamp"] = time_module.strftime("%Y-%m-%d %H:%M:%S")
                result["source"] = "Upload"
                st.session_state.transcription_history.append(result)
    else:
        # Display drag and drop instructions when no file is uploaded
        st.markdown(
            """
        ### 📁 Drag and drop an audio file here
        Supported formats: MP3, WAV, M4A, OGG, FLAC

        For best results:
        - Use files with clear speech and minimal background noise
        - Files should be under 10MB
        - Vietnamese audio works best
        """
        )
    st.markdown("</div>", unsafe_allow_html=True)


def record_tab(api_url, selected_model, selected_language):
    """Content for the Record Audio tab"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Record Audio")

    # Create columns for recording interface
    col1, col2 = st.columns([3, 1])

    with col1:
        # Show progress steps
        show_recording_steps(st.session_state.recording_step)

        # Display recording interface
        st.markdown("### 🎙️ Click the microphone to start/stop recording")

        # Handle recording
        audio_bytes = audio_recorder_component()

        if audio_bytes:
            st.session_state.current_recording = audio_bytes
            st.session_state.recording_step = 2

            # Review section
            st.markdown("### Review Your Recording")
            st.audio(audio_bytes, format="audio/wav")

            col_trans, col_retry = st.columns(2)
            with col_trans:
                if st.button("🎯 Transcribe", type="primary", key="transcribe_recording"):
                    st.session_state.recording_step = 3

                    # Process recording
                    result = process_recording(
                        audio_bytes, selected_model, selected_language, api_url
                    )

                    if result:
                        # Display the results
                        display_transcription_result(result, audio_bytes)

                        # Add to history
                        st.session_state.transcription_history.append(result)

            with col_retry:
                if st.button("🔄 Record Again", key="record_again"):
                    st.session_state.current_recording = None
                    st.session_state.recording_step = 1
                    st.rerun()

    with col2:
        # Display recording instructions
        recording_instructions()

    st.markdown("</div>", unsafe_allow_html=True)


def history_tab():
    """Content for the History tab"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Transcription History")

    # Display history and handle clear history action
    if display_history(st.session_state.transcription_history):
        st.session_state.transcription_history = []
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def system_status_tab(api_url, api_connected):
    """Content for the System Status tab"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("System Status and Metrics")

    col1, col2 = st.columns(2)

    with col1:
        # API Status
        display_api_status(api_url, api_connected)

    with col2:
        # Model Information
        display_model_info(api_url)

    # Link to metrics
    st.subheader("Detailed Metrics")
    st.info("View detailed system metrics and performance in the Grafana dashboard")

    if st.button("📊 Show Metrics Dashboard", key="show_metrics_btn"):
        st.session_state.show_metrics = True
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def metrics_dashboard(grafana_url):
    """Display the metrics dashboard"""
    st.header("System Metrics")
    st.markdown(f"[Open in Grafana]({grafana_url}/dashboards)")

    # Embed Grafana dashboard using iframe
    components.iframe(
        f"{grafana_url}/d/asr-dashboard/vietnamese-asr-dashboard?orgId=1&refresh=5s&kiosk",
        height=600,
    )


def setup_sidebar(api_url, api_connected):
    """Setup and display the sidebar"""
    with st.sidebar:
        # st.image("https://raw.githubusercontent.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc/main/ui/static/logo.png", width=100)
        st.header("Settings")

        # API Status display
        st.subheader("API Connection")
        if api_connected:
            st.success("✅ Connected to API")
        else:
            st.error("❌ Could not connect to API")
            st.info(f"Make sure the API is running at {api_url}")

        # Model selection
        st.subheader("Model Settings")
        models = get_available_models(api_url)

        # Create selectbox with formatted display
        model_display = st.selectbox(
            "Select Model",
            models,
            format_func=format_model_option,
            help="Choose the ASR model for transcription",
        )

        # Extract just the ID for API request
        selected_model = get_model_id(model_display)

        # Language selection
        languages = get_supported_languages(api_url)
        selected_language = st.selectbox(
            "Select Language",
            languages,
            format_func=lambda x: {
                "vi": "Vietnamese 🇻🇳",
                "en": "English 🇬🇧",
                "auto": "Auto-detect 🔍",
            }[x],
            help="Select the language of your audio",
        )

        # Theme selection
        st.subheader("UI Settings")
        theme = st.selectbox(
            "Theme",
            ["Light", "Dark"],
            index=0 if st.session_state.theme == "light" else 1,
            help="Select the UI theme",
        )

        if theme.lower() != st.session_state.theme:
            st.session_state.theme = theme.lower()
            # Apply theme changes (in a real implementation, this would update CSS)

        # Add metrics dashboard toggle
        st.subheader("Monitoring")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Metrics"):
                st.session_state.show_metrics = True
        with col2:
            if st.button("Hide Metrics"):
                st.session_state.show_metrics = False

        # GitHub link
        st.markdown("---")
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc)"
        )

        return selected_model, selected_language


def main():
    """Main application function"""
    # Initialize session state
    init_session_state()

    # Display custom header
    header()

    # Initialize API connection
    api_url = get_api_url()
    api_connected = check_api_status(api_url)

    # Setup sidebar and get selected settings
    selected_model, selected_language = setup_sidebar(api_url, api_connected)

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📤 Upload Audio", "🎙️ Record Audio", "📜 History", "⚙️ System Status"]
    )

    # Tab contents
    with tab1:
        upload_tab(api_url, selected_model, selected_language)

    with tab2:
        record_tab(api_url, selected_model, selected_language)

    with tab3:
        history_tab()

    with tab4:
        system_status_tab(api_url, api_connected)

    # Show metrics dashboard if toggled
    if st.session_state.show_metrics:
        metrics_dashboard(GRAFANA_URL)

    # Display footer
    footer()


if __name__ == "__main__":
    main()
