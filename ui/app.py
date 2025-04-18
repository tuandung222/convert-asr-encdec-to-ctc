import streamlit as st
import requests
import os
import io
import time
import tempfile
from pydub import AudioSegment
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
import streamlit.components.v1 as components

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
            return response.json()["models"]
        else:
            st.error(f"Failed to get models: {response.text}")
            return ["whisper-base"]
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return ["whisper-base"]

def get_supported_languages():
    try:
        response = requests.get(f"{API_URL}/languages")
        if response.status_code == 200:
            return response.json()["languages"]
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

def transcribe_audio(audio_file, model, language):
    try:
        files = {"file": audio_file}
        data = {"model": model, "language": language}
        
        with st.spinner("Transcribing audio..."):
            response = requests.post(f"{API_URL}/transcribe", files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to transcribe: {e}")
        return None

def create_confidence_chart(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "green"}
            ]
        }
    ))
    fig.update_layout(height=250)
    return fig

def main():
    st.set_page_config(
        page_title="Vietnamese ASR",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
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
        selected_model = st.selectbox("Select Model", models)
        
        # Language selection
        languages = get_supported_languages()
        selected_language = st.selectbox(
            "Select Language", 
            languages,
            format_func=lambda x: {"vi": "Vietnamese", "en": "English", "auto": "Auto-detect"}[x]
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
    
    # Show metrics dashboard if toggled
    if st.session_state.show_metrics:
        st.header("System Metrics")
        st.markdown(f"[Open in Grafana]({GRAFANA_URL}/d/asr-dashboard/vietnamese-asr-dashboard?orgId=1&refresh=5s)")
        
        # Embed Grafana dashboard using iframe
        components.iframe(
            f"{GRAFANA_URL}/d/asr-dashboard/vietnamese-asr-dashboard?orgId=1&refresh=5s&kiosk",
            height=600
        )
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Upload Audio", "Record Audio", "History", "System Status"])
    
    # Tab 1: Upload Audio
    with tab1:
        st.header("Upload Audio File")
        uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg", "flac"])
        
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
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Text:** {result['transcription']}")
                        st.markdown(f"**Model:** {result['model']}")
                        st.markdown(f"**Language:** {result['language']}")
                        st.markdown(f"**Processing Time:** {result['processing_time']:.2f} seconds")
                        
                        # Add download button
                        st.download_button(
                            label="Download Transcription",
                            data=result['transcription'],
                            file_name=f"transcription_{int(time.time())}.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        # Display confidence gauge
                        st.plotly_chart(create_confidence_chart(result["confidence"]))
                    
                    # Add to history with timestamp
                    result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.transcription_history.append(result)
    
    # Tab 2: Record Audio
    with tab2:
        st.header("Record Audio")
        st.warning("Note: Browser microphone access is required for recording.")
        
        # Audio recording using Streamlit's native audio recorder
        audio_bytes = st.audio_recorder(
            "Click to record", 
            pause_threshold=2.0,
            sample_rate=16000
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("Transcribe Recorded Audio"):
                # Create a temporary file for the audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Open the temp file for sending to API
                    with open(tmp_file_path, "rb") as audio_file:
                        result = transcribe_audio(audio_file, selected_model, selected_language)
                    
                    if result and result.get("success", False):
                        # Display the transcription results
                        st.subheader("Transcription Result")
                        st.success("Transcription completed successfully!")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Text:** {result['transcription']}")
                            st.markdown(f"**Model:** {result['model']}")
                            st.markdown(f"**Language:** {result['language']}")
                            st.markdown(f"**Processing Time:** {result['processing_time']:.2f} seconds")
                            
                            # Add download button
                            st.download_button(
                                label="Download Transcription",
                                data=result['transcription'],
                                file_name=f"transcription_{int(time.time())}.txt",
                                mime="text/plain"
                            )
                        
                        with col2:
                            # Display confidence gauge
                            st.plotly_chart(create_confidence_chart(result["confidence"]))
                        
                        # Add to history with timestamp
                        result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        result["source"] = "Recording"
                        st.session_state.transcription_history.append(result)
                finally:
                    # Clean up the temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
    
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
                display_df[["timestamp", "model", "language", "confidence", "processing_time", "transcription"]],
                use_container_width=True
            )
            
            # Add clear history button
            if st.button("Clear History"):
                st.session_state.transcription_history = []
                st.experimental_rerun()
    
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
            st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("Vietnamese ASR powered by Whisper models | Built with Streamlit | Metrics by Prometheus & Grafana")

if __name__ == "__main__":
    main() 