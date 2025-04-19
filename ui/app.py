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
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")


# Functions to interact with the API
def get_available_models():
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            data = response.json()
            # Handle both cases - either a dictionary with 'models' key or direct list
            if isinstance(data, dict) and "models" in data:
                return data["models"]
            elif isinstance(data, list):
                return data
            else:
                st.warning(f"Unexpected model data format: {data}")
                return ["phowhisper-tiny-ctc"]
        else:
            st.error(f"Failed to get models: {response.text}")
            return ["phowhisper-tiny-ctc"]
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return ["phowhisper-tiny-ctc"]


def get_supported_languages():
    try:
        response = requests.get(f"{API_URL}/languages")
        if response.status_code == 200:
            data = response.json()
            # Handle both cases - either a dictionary with 'languages' key or direct list
            if isinstance(data, dict) and "languages" in data:
                return data["languages"]
            elif isinstance(data, list):
                return data
            else:
                st.warning(f"Unexpected language data format: {data}")
                return ["vi", "en", "auto"]
        else:
            st.error(f"Failed to get languages: {response.text}")
            return ["vi", "en", "auto"]
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return ["vi", "en", "auto"]


def check_api_status():
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            return True
        return False
    except:
        return False


def get_model_types():
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            data = response.json()
            # Check if MODEL_TYPES is in the response
            if "model_types" in data:
                return data["model_types"]
            else:
                return ["pytorch", "onnx"]  # Default types if not specified
        else:
            st.error(f"Failed to get model types: {response.text}")
            return ["pytorch", "onnx"]
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return ["pytorch", "onnx"]


def transcribe_audio(audio_file, model, language, model_type=None):
    try:
        # Display debug info
        st.write(f"Sending model={model}, language={language}, model_type={model_type} to API")

        files = {"file": audio_file}
        data = {"model": model, "language": language}
        if model_type:
            data["model_type"] = model_type

        with st.spinner("Transcribing audio..."):
            response = requests.post(f"{API_URL}/transcribe", files=files, data=data)

        st.write(f"Response status: {response.status_code}")
        if response.status_code != 200:
            st.write(f"Response content: {response.text}")

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

            return result
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to transcribe: {e}")
        return None


def create_confidence_chart(confidence):
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


def handle_recording():
    try:
        audio_bytes = audio_recorder(
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x"
        )
        if audio_bytes:
            # Check audio length
            audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
            duration_seconds = len(audio) / 1000
            
            if duration_seconds < 0.5:
                st.warning("⚠️ Recording too short. Please record at least 0.5 seconds.")
                return None
            if duration_seconds > 60:
                st.warning("⚠️ Recording too long. Please keep it under 60 seconds.")
                return None
                
            return audio_bytes
    except Exception as e:
        st.error(f"Recording error: {str(e)}")
        return None


def show_recording_steps(current_step):
    steps = [
        "Record your voice 🎙️",
        "Review the recording 👂",
        "Transcribe to text ✍️"
    ]
    
    for i, step in enumerate(steps, 1):
        if i < current_step:
            st.markdown(f"✅ {i}. {step}")
        elif i == current_step:
            st.markdown(f"🔵 {i}. {step}")
        else:
            st.markdown(f"⚪ {i}. {step}")


def main():
    st.set_page_config(
        page_title="Vietnamese ASR", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded"
    )

    st.title("🎙️ Vietnamese Automatic Speech Recognition")

    # Check API connection
    api_connected = check_api_status()

    # Sidebar for settings
    with st.sidebar:
        st.header("API Connection")
        if api_connected:
            st.success("✅ Connected to API")
        else:
            st.error("❌ Could not connect to API")
            st.info(f"Make sure the API is running at {API_URL}")

        st.header("Settings")

        # Model selection
        models = get_available_models()

        # Format model objects for display
        def format_model_option(model):
            if isinstance(model, dict) and "id" in model:
                return f"{model['name']} - {model.get('description', '')}"
            return model

        # Create selectbox with formatted display
        model_display = st.selectbox("Select Model", models, format_func=format_model_option)

        # Extract just the ID for API request
        if isinstance(model_display, dict) and "id" in model_display:
            selected_model = model_display["id"]
        else:
            selected_model = model_display

        # Language selection
        languages = get_supported_languages()
        selected_language = st.selectbox(
            "Select Language",
            languages,
            format_func=lambda x: {"vi": "Vietnamese", "en": "English", "auto": "Auto-detect"}[x],
        )

        # Add metrics dashboard link
        st.header("Monitoring")
        if st.button("View Metrics Dashboard"):
            st.session_state.show_metrics = True

        if st.button("Hide Metrics Dashboard"):
            st.session_state.show_metrics = False

    # Initialize session state for transcription history and metrics toggle
    if "transcription_history" not in st.session_state:
        st.session_state.transcription_history = []

    if "show_metrics" not in st.session_state:
        st.session_state.show_metrics = False

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Upload Audio", "Record Audio", "History", "System Status"])

    # Tab 1: Upload Audio
    with tab1:
        st.header("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file", type=["mp3", "wav", "m4a", "ogg", "flac"]
        )

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")

            if st.button("Transcribe Uploaded Audio"):
                # Reset file position
                uploaded_file.seek(0)

                # Process the transcription
                result = transcribe_audio(uploaded_file, selected_model, selected_language)

                if result and result.get("success", False):
                    # Display the transcription results
                    st.subheader("Transcription Result")
                    st.success("Transcription completed successfully!")

                    # Display transcription in full width instead of columns
                    st.markdown(f"**Text:** {result['transcription']}")
                    st.markdown(f"**Model:** {result['model']}")
                    st.markdown(f"**Language:** {result['language']}")
                    st.markdown(f"**Processing Time:** {result['processing_time']:.2f} seconds")

                    # Add download button
                    st.download_button(
                        label="Download Transcription",
                        data=result["transcription"],
                        file_name=f"transcription_{time_module.strftime('%Y-%m-%d %H:%M:%S')}.txt",
                        mime="text/plain",
                    )

                    # Add to history with timestamp
                    result["timestamp"] = time_module.strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.transcription_history.append(result)

    # Tab 2: Record Audio
    with tab2:
        st.header("Record Audio")

        # Initialize recording state
        if "recording_step" not in st.session_state:
            st.session_state.recording_step = 1
        if "current_recording" not in st.session_state:
            st.session_state.current_recording = None

        col1, col2 = st.columns([3, 1])

        with col1:
            # Show progress steps
            show_recording_steps(st.session_state.recording_step)
            
            st.markdown("### 🎙️ Click the microphone to start/stop recording")
            
            # Handle recording
            audio_bytes = handle_recording()
            
            if audio_bytes:
                st.session_state.current_recording = audio_bytes
                st.session_state.recording_step = 2
                
                # Review section
                st.markdown("### Review Your Recording")
                st.audio(audio_bytes, format="audio/wav")
                
                col_trans, col_retry = st.columns(2)
                with col_trans:
                    if st.button("🎯 Transcribe", key="transcribe_recording"):
                        st.session_state.recording_step = 3
                        
                        # Create a temporary file for the audio
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_file_path = tmp_file.name

                        try:
                            # Open the temp file for sending to API
                            with open(tmp_file_path, "rb") as audio_file:
                                result = transcribe_audio(audio_file, selected_model, selected_language)

                            if result and result.get("success", False):
                                # Display the transcription results in a styled container
                                st.success("✓ Transcription complete")

                                # Result container with styling
                                with st.container():
                                    st.markdown("### Transcription Result")
                                    st.markdown(f"**{result['transcription']}**")

                                    # Display metrics in columns
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Duration", f"{result.get('duration', 0):.2f}s")
                                    with col2:
                                        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                                    with col3:
                                        st.metric("Real-time Factor", f"{result.get('real_time_factor', 0):.2f}x")

                                # Add download buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        label="Download Transcription",
                                        data=result["transcription"],
                                        file_name=f"transcription_{time_module.strftime('%Y-%m-%d %H:%M:%S')}.txt",
                                        mime="text/plain",
                                    )
                                with col2:
                                    st.download_button(
                                        label="Download Audio",
                                        data=audio_bytes,
                                        file_name=f"recording_{time_module.strftime('%Y-%m-%d %H:%M:%S')}.wav",
                                        mime="audio/wav",
                                    )

                                # Add to history with timestamp
                                result["timestamp"] = time_module.strftime("%Y-%m-%d %H:%M:%S")
                                result["source"] = "Recording"
                                st.session_state.transcription_history.append(result)
                        finally:
                            # Clean up the temporary file
                            if os.path.exists(tmp_file_path):
                                os.unlink(tmp_file_path)
                
                with col_retry:
                    if st.button("🔄 Record Again", key="record_again"):
                        st.session_state.current_recording = None
                        st.session_state.recording_step = 1
                        st.rerun()

        with col2:
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

    # Tab 3: Transcription History
    with tab3:
        st.header("Transcription History")

        if not st.session_state.transcription_history:
            st.info("No transcriptions yet. Upload or record audio to get started.")
        else:
            # Create a dataframe from the history
            history_df = pd.DataFrame(st.session_state.transcription_history)

            # Format the dataframe for display
            display_df = history_df.copy()
            if "timestamp" in display_df.columns:
                display_df = display_df.sort_values("timestamp", ascending=False)

            # Display the history as a dataframe
            st.dataframe(
                display_df[["timestamp", "model", "language", "processing_time", "transcription"]],
                use_container_width=True,
            )

            # Add clear history button
            if st.button("Clear History"):
                st.session_state.transcription_history = []
                st.rerun()

    # Tab 4: System Status
    with tab4:
        st.header("System Status and Metrics")

        col1, col2 = st.columns(2)

        with col1:
            # API Status
            st.subheader("API Status")
            if api_connected:
                try:
                    response = requests.get(f"{API_URL}/health")
                    if response.status_code == 200:
                        health_data = response.json()
                        st.success("✅ API is healthy")
                        st.json(health_data)
                    else:
                        st.error(f"❌ API health check failed: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Cannot connect to API: {e}")
            else:
                st.error("❌ API is not connected")

        with col2:
            # Model Information
            st.subheader("Available Models")
            try:
                response = requests.get(f"{API_URL}/models")
                if response.status_code == 200:
                    models_data = response.json()
                    for model in models_data:
                        st.markdown(f"**{model['id']}**: {model['description']}")
                else:
                    st.error(f"❌ Could not retrieve model information")
            except Exception as e:
                st.error(f"❌ Error getting model information: {e}")

        # Link to metrics
        st.subheader("Detailed Metrics")
        st.info("View detailed system metrics and performance in the Grafana dashboard")

        if st.button("Show Metrics Dashboard", key="show_metrics_btn"):
            st.session_state.show_metrics = True
            st.rerun()

    # Show metrics dashboard if toggled
    if st.session_state.show_metrics:
        st.header("System Metrics")
        st.markdown(f"[Open in Grafana]({GRAFANA_URL}/dashboards)")

        # Embed Grafana dashboard using iframe
        components.iframe(
            f"{GRAFANA_URL}/d/asr-dashboard/vietnamese-asr-dashboard?orgId=1&refresh=5s&kiosk",
            height=600,
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "Vietnamese ASR powered by Whisper models | Built with Streamlit | Metrics by Prometheus & Grafana"
    )


if __name__ == "__main__":
    main()
