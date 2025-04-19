#!/usr/bin/env python3
"""
Script to test ASR model with a real Vietnamese audio example.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vietnamese-test")

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.inference_model import create_asr_model


def test_vietnamese_example(
    audio_path: str = "examples/vietnamese_example.mp3",
    model_id: str = "tuandunghcmut/PhoWhisper-tiny-CTC",
    model_type: str = "onnx",  # Use ONNX by default
    device: str = "cpu",
    verbose: bool = False,
):
    """
    Test ASR model with a Vietnamese audio example.

    Args:
        audio_path: Path to the audio file
        model_id: Model ID or path
        model_type: Model type ('pytorch' or 'onnx')
        device: Device to run inference on ('cpu' or 'cuda')
        verbose: Whether to show verbose output
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing model on Vietnamese audio example")
    logger.info(f"{'='*50}")
    logger.info(f"Model: {model_id}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Audio: {audio_path}")
    logger.info(f"Device: {device}")
    logger.info(f"{'='*50}\n")

    if not os.path.exists(audio_path):
        logger.error(f"Error: Audio file not found at {audio_path}")
        return

    # Make sure checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)

    try:
        # Create the ASR model
        logger.info("Loading ASR model...")
        model = create_asr_model(model_id, device, model_type=model_type)

        # Run inference
        logger.info(f"Running inference on {audio_path}...")
        start_time = time.time()
        result = model.transcribe(audio_path)
        inference_time = time.time() - start_time

        predicted_text = result["text"]
        audio_duration = result.get("duration", 0.0)
        real_time_factor = inference_time / max(1.0, audio_duration)

        logger.info("\nResults:")
        logger.info(f"Transcription: {predicted_text}")
        logger.info(f"Audio duration: {audio_duration:.2f}s")
        logger.info(f"Inference time: {inference_time:.2f}s")
        logger.info(f"Real-time factor: {real_time_factor:.2f}x")

        # Any debug info available?
        if "debug_info" in result and verbose:
            logger.info("\nDebug information:")
            debug_info = result["debug_info"]
            if debug_info:
                for t, timestamp_info in enumerate(debug_info[:10]):  # Show first 10 timestamps
                    logger.info(f"Timestamp {t}:")
                    for token_id, token_text, prob in timestamp_info:
                        logger.info(f"  {token_id} ({token_text}): {prob:.4f}")

        return {
            "transcription": predicted_text,
            "audio_duration": audio_duration,
            "inference_time": inference_time,
            "real_time_factor": real_time_factor,
        }

    except Exception as e:
        logger.error(f"Error during model testing: {str(e)}")
        # Print stack trace for debugging
        import traceback

        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ASR model with Vietnamese example audio")
    parser.add_argument(
        "--audio", default="examples/vietnamese_example.mp3", help="Path to audio file"
    )
    parser.add_argument(
        "--model", default="tuandunghcmut/PhoWhisper-tiny-CTC", help="Model ID or path"
    )
    parser.add_argument("--type", default="onnx", choices=["pytorch", "onnx"], help="Model type")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--verbose", action="store_true", help="Show more details")

    args = parser.parse_args()

    test_vietnamese_example(args.audio, args.model, args.type, args.device, args.verbose)
