# CURRENTLY NOT USED
# import os
# import sys
# import time
# import argparse
# import gradio as gr
# import torch
# import numpy as np
# import logging
# from typing import Tuple, Dict, Any, Optional, List

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# # Add src to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # Import model
# try:
#     from src.models.inference_model import create_asr_model, ASRInferenceModel
# except ImportError:
#     logger.error("Failed to import inference model. Make sure the project structure is correct.")
#     raise

# # Constants
# MODEL_NAME = "tuandunghcmut/PhoWhisper-tiny-CTC"
# DEFAULT_LANGUAGE = "vi"
# SAMPLE_RATE = 16000
# MAX_AUDIO_LENGTH = 30  # in seconds

# def load_model(device: str = "cpu") -> ASRInferenceModel:
#     """Load the ASR model for inference"""
#     logger.info(f"Loading ASR model {MODEL_NAME} on {device}")
#     try:
#         model = create_asr_model(MODEL_NAME, device)
#         logger.info("Model loaded successfully")
#         return model
#     except Exception as e:
#         logger.error(f"Error loading model: {e}")
#         raise

# def process_audio(
#     audio_path: str,
#     model: ASRInferenceModel
# ) -> Dict[str, Any]:
#     """Process audio and return transcription results"""
#     start_time = time.time()

#     try:
#         # Transcribe audio
#         result = model.transcribe(audio_path)

#         # Calculate processing time
#         processing_time = time.time() - start_time
#         real_time_factor = processing_time / (result.get("duration", 1))

#         # Add performance metrics
#         result["processing_time"] = f"{processing_time:.2f}s"
#         result["real_time_factor"] = f"{real_time_factor:.2f}x"

#         return result
#     except Exception as e:
#         logger.error(f"Error processing audio: {e}")
#         return {
#             "text": f"Error: {str(e)}",
#             "processing_time": "N/A",
#             "real_time_factor": "N/A",
#             "error": True
#         }

# def transcribe_audio(
#     audio,
#     model: ASRInferenceModel,
#     language: str = DEFAULT_LANGUAGE
# ) -> Tuple[str, Dict[str, Any]]:
#     """Transcribe audio using the loaded model"""
#     if audio is None:
#         return "No audio provided. Please upload or record audio.", {}

#     # Save temporary audio file
#     temp_path = "temp_audio.wav"

#     # Handle different audio input formats
#     if isinstance(audio, tuple):  # From audio recorder
#         sample_rate, audio_data = audio
#         # Save as wav file for processing
#         import soundfile as sf
#         sf.write(temp_path, audio_data, sample_rate)
#     else:  # From file upload
#         temp_path = audio  # Use the uploaded file path directly

#     # Process with model
#     result = process_audio(temp_path, model)

#     # Extract main transcription and detailed info
#     transcription = result.get("text", "Error transcribing audio")
#     details = {
#         "Duration": result.get("duration", "N/A"),
#         "Processing Time": result.get("processing_time", "N/A"),
#         "Real-time Factor": result.get("real_time_factor", "N/A"),
#     }

#     # Clean up temp file if needed
#     if isinstance(audio, tuple) and os.path.exists(temp_path):
#         os.remove(temp_path)

#     return transcription, details

# def create_gradio_app(model: ASRInferenceModel) -> gr.Blocks:
#     """Create and return the Gradio interface"""
#     with gr.Blocks(title="Vietnamese ASR with PhoWhisper-CTC") as app:
#         gr.Markdown("# Vietnamese Automatic Speech Recognition")
#         gr.Markdown("## Using CTC-based PhoWhisper Model")

#         with gr.Row():
#             with gr.Column(scale=1):
#                 # Input methods
#                 gr.Markdown("### Input Audio")
#                 audio_input = gr.Audio(
#                     sources=["upload", "microphone"],
#                     type="filepath",
#                     label="Upload or Record Audio"
#                 )
#                 language = gr.Dropdown(
#                     choices=["vi"],
#                     value="vi",
#                     label="Language",
#                     interactive=False
#                 )
#                 transcribe_btn = gr.Button("Transcribe", variant="primary")

#                 # Model info
#                 with gr.Accordion("Model Information", open=False):
#                     gr.Markdown(f"""
#                     - **Model**: PhoWhisper-tiny-CTC
#                     - **Architecture**: CTC-based speech recognition
#                     - **Vocabulary**: Vietnamese
#                     - **Device**: {model.device}
#                     - **Source**: [HuggingFace](https://huggingface.co/{MODEL_NAME})
#                     """)

#             with gr.Column(scale=1):
#                 # Output
#                 gr.Markdown("### Transcription Result")
#                 text_output = gr.Textbox(label="Transcription", lines=5)
#                 details_output = gr.JSON(label="Details")

#                 # Examples
#                 with gr.Accordion("Examples", open=True):
#                     gr.Examples(
#                         examples=[
#                             ["examples/example1.wav"],
#                             ["examples/example2.wav"],
#                         ],
#                         inputs=[audio_input],
#                         outputs=[text_output, details_output],
#                         fn=lambda a: transcribe_audio(a, model),
#                         cache_examples=True,
#                     )

#         # Footer
#         gr.Markdown("---")
#         gr.Markdown("Created by: Tuan Dung. [GitHub Repository](https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc)")

#         # Set up event handlers
#         transcribe_btn.click(
#             fn=transcribe_audio,
#             inputs=[audio_input, model, language],
#             outputs=[text_output, details_output],
#         )

#     return app

# def main():
#     """Main entry point for the Gradio app"""
#     parser = argparse.ArgumentParser(description="Vietnamese ASR Gradio Demo")
#     parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda)")
#     parser.add_argument("--share", action="store_true", help="Share the app publicly")
#     parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
#     args = parser.parse_args()

#     # Check if CUDA is available when requested
#     if args.device == "cuda" and not torch.cuda.is_available():
#         logger.warning("CUDA requested but not available. Falling back to CPU.")
#         args.device = "cpu"

#     # Create examples directory if it doesn't exist
#     os.makedirs("examples", exist_ok=True)

#     # Load model
#     model = load_model(args.device)

#     # Create and launch Gradio app
#     app = create_gradio_app(model)
#     app.launch(
#         server_name="0.0.0.0",
#         server_port=args.port,
#         share=args.share,
#         debug=True
#     )

# if __name__ == "__main__":
#     main()
