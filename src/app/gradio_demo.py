import os
import sys
import tempfile
import time

import gradio as gr
import numpy as np
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import our ASR model wrapper
from src.models.inference_model import create_asr_model


def load_model(device="cpu"):
    """Load the ASR model"""
    return create_asr_model(
        model_type="pytorch",
        model_name="vinai/PhoWhisper-tiny",
        repo_id="tuandunghcmut/PhoWhisper-tiny-CTC",
        checkpoint_filename="best-val_wer=0.3986.ckpt",
        use_cuda=(device == "cuda"),
    )


def transcribe_audio(audio_path, model, progress=gr.Progress()):
    """Transcribe audio file using the model"""
    if audio_path is None:
        return "Please upload or record audio first."

    try:
        progress(0.1, "Loading audio...")

        # Check if the audio is a tuple (from Gradio microphone)
        if isinstance(audio_path, tuple):
            # For microphone input, it's (sample_rate, audio_data)
            sample_rate, audio_data = audio_path
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # Save the audio data as a WAV file
                import soundfile as sf

                sf.write(temp_file.name, audio_data, sample_rate)
                temp_path = temp_file.name

            try:
                progress(0.3, "Processing audio...")
                result = model.transcribe(temp_path)
                progress(0.9, "Finalizing...")
                # Get transcription from the result
                transcription = result.get("text", "")
                if "transcription" in result:
                    transcription = result["transcription"]
                return transcription
            finally:
                # Clean up the temporary file
                os.unlink(temp_path)
        else:
            # For uploaded files, we have a path
            progress(0.3, "Processing audio...")
            result = model.transcribe(audio_path)
            progress(0.9, "Finalizing...")
            # Get transcription from the result
            transcription = result.get("text", "")
            if "transcription" in result:
                transcription = result["transcription"]
            return transcription
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"


def create_demo(share=False, device="cpu"):
    """Create the Gradio demo interface"""
    # Load the model
    model = load_model(device)

    # Define the interface
    demo = gr.Interface(
        fn=lambda audio_path, progress=None: transcribe_audio(audio_path, model, progress),
        inputs=[gr.Audio(sources=["upload", "microphone"], type="filepath")],
        outputs=[gr.Textbox(label="Transcription", lines=5)],
        title="Vietnamese Automatic Speech Recognition",
        description="""Upload an audio file or record your voice to transcribe Vietnamese speech.

This demo uses a PhoWhisper-CTC model trained on Vietnamese speech data.
The model automatically handles Vietnamese speech recognition without the need for language selection.""",
        article="""
## About the Model

This demo uses a CTC-based speech recognition model derived from PhoWhisper,
optimized for Vietnamese speech recognition.

- Model: PhoWhisper-tiny-CTC
- Word Error Rate (WER): 41% on the VietBud500 test set
- Inference speed: Real-time (more than 2x faster than speech duration)

## Tips for Best Results

- Speak clearly with minimal background noise
- For recording, use a good microphone and quiet environment
- For uploaded files, ensure they are clear audio recordings
        """,
        examples=[["examples/example1.wav"], ["examples/example2.wav"]],
        cache_examples=True,
        theme=gr.themes.Soft(),
    )

    return demo


def main():
    """Run the Gradio demo"""
    # Get arguments from environment or use defaults
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    device = os.environ.get("INFERENCE_DEVICE", "cpu")
    port = int(os.environ.get("PORT", 7860))

    # Create and launch the demo
    demo = create_demo(share=share, device=device)
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)


if __name__ == "__main__":
    main()
