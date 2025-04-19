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

def transcribe_audio(audio_file, model, language):
    try:
        # Log file information for debugging
        st.write(f"File type: {type(audio_file)}")
        if hasattr(audio_file, 'name'):
            st.write(f"File name: {audio_file.name}")
        if hasattr(audio_file, 'type'):
            st.write(f"File content type: {audio_file.type}")
        
        files = {"file": audio_file}
        data = {"model": model, "language": language}
        
        st.write(f"Sending request to {API_URL}/transcribe with model={model}, language={language}")
        
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
            # Map 'text' field to 'transcription' for UI consistency
            if "text" in result and "transcription" not in result:
                result["transcription"] = result["text"]
            return result
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to transcribe: {e}")
        import traceback
        st.error(traceback.format_exc())
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
        # Create a format function to display model names but store model IDs
        def format_model(model):
            if isinstance(model, dict) and 'id' in model:
                return f"{model['name']} - {model['description']}"
            else:
                return model
        
        # Use a selectbox with the format function
        selected_model_display = st.selectbox(
            "Select Model", 
            models,
            format_func=format_model
        )
        
        # Extract just the model ID for API calls
        if isinstance(selected_model_display, dict) and 'id' in selected_model_display:
            selected_model = selected_model_display['id']
        else:
            selected_model = selected_model_display
        
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
        
        # Initialize recording state in session_state
        if "recording_state" not in st.session_state:
            st.session_state.recording_state = "idle"  # States: idle, recording, recorded
        
        if "audio_bytes" not in st.session_state:
            st.session_state.audio_bytes = None
            
        # Show different UI based on recording state
        if st.session_state.recording_state == "idle":
            st.info("Click 'Start Recording' to begin. Ensure your microphone is enabled.")
            
            # UI for starting recording
            start_col1, start_col2 = st.columns([1, 3])
            with start_col1:
                if st.button("🎙️ Start Recording"):
                    st.session_state.recording_state = "recording"
                    st.session_state.audio_buffer = []
                    st.rerun()
            
            with start_col2:
                st.markdown("**Microphone access is required. Please allow it when prompted by your browser.**")
                
            # Show a sample of what to expect
            st.markdown("---")
            st.markdown("#### Example Result")
            st.markdown("After recording, you'll see your audio waveform and can transcribe it.")
            st.image("https://miro.medium.com/max/1400/1*wMSKA7jf7gFiSb-2nVNRlg.png", width=400)
            
        elif st.session_state.recording_state == "recording":
            # UI during recording
            st.warning("🔴 Recording in progress...")
            
            # Display a timer
            placeholder = st.empty()
            
            # Try to setup and use streamlit-webrtc for recording
            try:
                from streamlit_webrtc import webrtc_streamer
                import av
                import time
                
                # Initialize or get duration counter
                if "recording_start_time" not in st.session_state:
                    st.session_state.recording_start_time = time.time()
                
                # Show recording duration
                elapsed = time.time() - st.session_state.recording_start_time
                placeholder.markdown(f"### Recording for: {int(elapsed)} seconds")
                
                # Setup a container for audio recording
                audio_buffer = st.session_state.audio_buffer
                
                def audio_callback(frame):
                    audio_buffer.append(frame.to_ndarray())
                    return frame
                
                # Create webrtc streamer
                webrtc_ctx = webrtc_streamer(
                    key="audio-recorder",
                    audio_receiver_size=1024,
                    media_stream_constraints={"video": False, "audio": True},
                    video_processor_factory=None,
                    audio_processor_factory=lambda: audio_callback,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    # This helps ensure compatibility across browsers
                )
                
                # Check if webrtc has been stopped - consider recording finished
                if webrtc_ctx.state.playing == False and len(audio_buffer) > 0:
                    # Process the recorded audio
                    try:
                        import numpy as np
                        from pydub import AudioSegment
                        import io
                        
                        # Concatenate all audio frames
                        audio_frames = np.concatenate(audio_buffer, axis=0)
                        
                        # Convert to int16 format
                        audio_frames = (audio_frames * 32767).astype(np.int16)
                        
                        # Create AudioSegment
                        audio_segment = AudioSegment(
                            audio_frames.tobytes(),
                            frame_rate=16000,
                            sample_width=2,
                            channels=1
                        )
                        
                        # Export to WAV bytes
                        buffer = io.BytesIO()
                        audio_segment.export(buffer, format="wav")
                        st.session_state.audio_bytes = buffer.getvalue()
                        
                        # Change state to recorded
                        st.session_state.recording_state = "recorded"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to process audio: {e}")
                        st.session_state.recording_state = "idle"
                        st.rerun()
                
                # Stop recording button
                stop_col1, stop_col2 = st.columns([1, 3])
                with stop_col1:
                    if st.button("⏹️ Stop Recording"):
                        st.warning("Stopping recording... Please wait.")
                        st.session_state.recording_state = "processing"
                        # The actual processing will happen on the next rerun when webrtc_ctx.state.playing is False
                        st.rerun()
                        
                with stop_col2:
                    st.markdown("**Click 'Stop Recording' when you're finished speaking.**")
                
            except ImportError:
                st.error("Please install required packages: `pip install streamlit-webrtc av pydub`")
                st.info("Alternatively, you can use the file upload tab to upload audio files.")
                # Reset state on error
                st.session_state.recording_state = "idle"
                
            except Exception as e:
                st.error(f"Error during recording: {e}")
                import traceback
                st.error(traceback.format_exc())
                # Reset state on error
                st.session_state.recording_state = "idle"
                
        elif st.session_state.recording_state == "recorded":
            # UI after recording is complete
            st.success("✅ Recording completed!")
            
            # Display the recorded audio
            if st.session_state.audio_bytes:
                st.audio(st.session_state.audio_bytes, format="audio/wav")
                
                # Buttons for actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Record Again"):
                        st.session_state.recording_state = "idle"
                        st.session_state.audio_bytes = None
                        st.rerun()
                
                with col2:
                    if st.button("🎯 Transcribe"):
                        # Create a temporary file for the audio
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(st.session_state.audio_bytes)
                            tmp_file_path = tmp_file.name
                        
                        try:
                            # Open the temp file for sending to API
                            with open(tmp_file_path, "rb") as audio_file:
                                # Set a filename for the file to ensure it's properly processed by FastAPI
                                class NamedBytesIO:
                                    def __init__(self, file, name):
                                        self.file = file
                                        self.name = name
                                    
                                    def read(self, *args, **kwargs):
                                        return self.file.read(*args, **kwargs)
                                    
                                    def seek(self, *args, **kwargs):
                                        return self.file.seek(*args, **kwargs)
                                
                                named_file = NamedBytesIO(audio_file, "recorded_audio.wav")
                                
                                # Show transcription progress
                                with st.spinner("Transcribing... Please wait."):
                                    result = transcribe_audio(named_file, selected_model, selected_language)
                                
                                if result and result.get("success", False):
                                    # Store result in session state
                                    st.session_state.transcription_result = result
                                    
                                    # Add to history with timestamp
                                    result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                                    result["source"] = "Recording"
                                    st.session_state.transcription_history.append(result)
                                    
                                    # Switch to result display
                                    st.session_state.recording_state = "result"
                                    st.rerun()
                                else:
                                    st.error("Transcription failed. Please try again.")
                            
                        except Exception as e:
                            st.error(f"Error during transcription: {e}")
                            import traceback
                            st.error(traceback.format_exc())
                        finally:
                            # Clean up the temporary file
                            if os.path.exists(tmp_file_path):
                                os.unlink(tmp_file_path)
                
                with col3:
                    # Download button for the recorded audio
                    st.download_button(
                        label="💾 Download Audio",
                        data=st.session_state.audio_bytes,
                        file_name=f"recording_{int(time.time())}.wav",
                        mime="audio/wav"
                    )
            else:
                st.error("No audio data found. Please record again.")
                st.session_state.recording_state = "idle"
                st.rerun()
                
        elif st.session_state.recording_state == "result":
            # UI for displaying transcription result
            if "transcription_result" in st.session_state:
                result = st.session_state.transcription_result
                
                # Display the recorded audio
                if st.session_state.audio_bytes:
                    st.audio(st.session_state.audio_bytes, format="audio/wav")
                
                # Display the transcription results
                st.subheader("Transcription Result")
                st.success("Transcription completed successfully!")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Text:** {result['transcription']}")
                    st.markdown(f"**Model:** {result['model']}")
                    st.markdown(f"**Language:** {result['language']}")
                    st.markdown(f"**Processing Time:** {result['processing_time']:.2f} seconds")
                    
                    # Add download button for text
                    st.download_button(
                        label="Download Transcription",
                        data=result['transcription'],
                        file_name=f"transcription_{int(time.time())}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Display confidence gauge
                    st.plotly_chart(create_confidence_chart(result["confidence"]))
                
                # Action buttons
                new_col1, new_col2 = st.columns(2)
                
                with new_col1:
                    if st.button("🔄 Record New Audio"):
                        st.session_state.recording_state = "idle"
                        st.session_state.audio_bytes = None
                        if "transcription_result" in st.session_state:
                            del st.session_state.transcription_result
                        st.rerun()
                
                with new_col2:
                    # Download button for the recorded audio
                    st.download_button(
                        label="💾 Download Audio",
                        data=st.session_state.audio_bytes,
                        file_name=f"recording_{int(time.time())}.wav",
                        mime="audio/wav",
                        key="download_audio_result"
                    )
            else:
                st.error("No transcription result found. Please try again.")
                st.session_state.recording_state = "idle"
                st.rerun()
    
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
    
    # Footer
    st.markdown("---")
    st.markdown("Vietnamese ASR powered by Whisper models | Built with Streamlit | Metrics by Prometheus & Grafana")

if __name__ == "__main__":
    main() 