#!/usr/bin/env python
"""
Test script for the Vietnamese ASR model.
This script loads the model and runs inference on a sample audio file.
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import the model
try:
    from src.models.inference_model import create_asr_model
except ImportError:
    logger.error("Failed to import inference model. Make sure the project structure is correct.")
    raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Vietnamese ASR model")
    parser.add_argument("--audio", type=str, default="examples/example1.wav",
                       help="Path to audio file or directory")
    parser.add_argument("--model", type=str, 
                       default="tuandunghcmut/PhoWhisper-tiny-CTC",
                       help="Model to use for inference")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run inference on (cpu or cuda)")
    return parser.parse_args()

def process_audio(audio_path: str, model_id: str, device: str) -> Dict[str, Any]:
    """Process an audio file and return transcription results."""
    try:
        # Create the model
        logger.info(f"Loading model {model_id} on {device}")
        model = create_asr_model(model_id, device)
        logger.info("Model loaded successfully")
        
        # Transcribe the audio
        logger.info(f"Transcribing {audio_path}")
        start_time = time.time()
        result = model.transcribe(audio_path)
        processing_time = time.time() - start_time
        
        # Add processing time information
        result["processing_time"] = f"{processing_time:.2f}s"
        if "duration" in result:
            rtf = processing_time / result["duration"]
            result["real_time_factor"] = f"{rtf:.2f}x"
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return {"error": str(e)}

def main():
    """Main entry point."""
    args = parse_args()
    
    # Check if the audio file exists
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        return 1
    
    # Process the audio
    result = process_audio(args.audio, args.model, args.device)
    
    # Print the results
    if "error" in result:
        logger.error(f"Transcription failed: {result['error']}")
        return 1
    
    logger.info("=" * 80)
    logger.info(f"Transcription: {result.get('text', '')}")
    logger.info(f"Duration: {result.get('duration', 'N/A')}s")
    logger.info(f"Processing time: {result.get('processing_time', 'N/A')}")
    if "real_time_factor" in result:
        logger.info(f"Real-time factor: {result['real_time_factor']}")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 