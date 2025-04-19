import argparse
import os

import torch

from src.models.inference_model import create_asr_model


def test_model_loading(model_id="tuandunghcmut/PhoWhisper-tiny-CTC", device="cpu"):
    """Test model loading and inference"""
    print(f"Testing model loading: {model_id} on {device}")

    # Make sure checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)

    # Create the ASR model
    model = create_asr_model(model_id, device)

    # Test inference on a silent audio file
    print("Testing inference on test audio file")
    result = model.transcribe("examples/test_audio.wav")

    print("Model test results:")
    print(f"Transcription: {result['text']}")
    print(f"Duration: {result['duration']:.2f}s")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PhoWhisper CTC model loading and inference")
    parser.add_argument(
        "--model", default="tuandunghcmut/PhoWhisper-tiny-CTC", help="Model ID or path"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")

    args = parser.parse_args()
    test_model_loading(args.model, args.device)
