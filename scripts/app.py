#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import gradio as gr
import numpy as np
from pathlib import Path
import torch

# Add parent directory to path to import source modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.models.model_manager import ModelManager
from source.utils.logging import setup_logging, log_system_info
from source.utils.config import setup_config_for_inference

# Set up logger
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Gradio app for Vietnamese ASR CTC model")
    
    parser.add_argument(
        "--model_config", 
        type=str, 
        default="configs/model_config.yaml", 
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--inference_config", 
        type=str, 
        default="configs/inference_config.yaml", 
        help="Path to inference configuration file"
    )
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to model checkpoint (overrides config)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="Device to run inference on (cpu or cuda)"
    )
    
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        help="Logging level"
    )
    
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Share the app publicly through Gradio"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Port to run the app on"
    )
    
    return parser.parse_args()

def create_model_manager(args):
    """Create a model manager from arguments."""
    # Set up configuration
    override_values = {}
    if args.checkpoint:
        override_values["model.checkpoint_path"] = args.checkpoint
    
    override_values["inference.device"] = args.device
    
    config = setup_config_for_inference(
        model_config_path=args.model_config,
        inference_config_path=args.inference_config,
        override_values=override_values
    )
    
    # Create model manager
    model_manager = ModelManager(config)
    
    return model_manager

def transcribe_audio(model_manager, audio_file_path=None, audio_sample_rate=None, audio_data=None):
    """
    Transcribe audio using the model manager.
    
    Args:
        model_manager: Model manager instance
        audio_file_path: Path to audio file (if provided)
        audio_sample_rate: Sample rate of audio data (if provided)
        audio_data: Audio data as array (if provided)
        
    Returns:
        Transcription text
    """
    try:
        if audio_file_path:
            # Transcribe from file path
            return model_manager.transcribe_file(audio_file_path)
        elif audio_data is not None and audio_sample_rate:
            # Transcribe from audio data
            return model_manager.transcribe_audio(audio_data)
        else:
            return "Error: No audio provided"
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return f"Error: {str(e)}"

def main():
    """Main function to run the Gradio app."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level, log_file="logs/app.log")
    
    # Log system information
    log_system_info()
    
    # Create model manager
    model_manager = create_model_manager(args)
    
    # Define Gradio interface
    def process_audio(audio):
        if audio is None:
            return "No audio provided."
        
        try:
            audio_path, sample_rate = audio
            if sample_rate != 16000:
                # Gradio handles resampling automatically
                # For future customization, you could implement resampling here
                pass
            
            # Get audio numpy array from the tuple
            audio_data = audio[1]
            
            # Transcribe audio
            transcription = model_manager.transcribe_audio(audio_data)
            return transcription
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return f"Error: {str(e)}"
    
    # Create Gradio interface
    demo = gr.Interface(
        fn=process_audio,
        inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio Input"),
        outputs=gr.Textbox(label="Transcription"),
        title="Vietnamese Speech Recognition with CTC",
        description="""
        This demo transcribes Vietnamese speech to text using a CTC-based model.
        You can either record audio using your microphone or upload an audio file.
        The model works best with clear speech and minimal background noise.
        """,
        examples=[
            ["examples/example1.wav"],
            ["examples/example2.wav"],
        ],
        allow_flagging="never",
    )
    
    # Create examples directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )

if __name__ == "__main__":
    main() 