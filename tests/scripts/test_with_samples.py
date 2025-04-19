import json
import os

import torch

from src.models.inference_model import create_asr_model


def test_model_with_samples(
    model_id="tuandunghcmut/PhoWhisper-tiny-CTC", samples_dir="examples/viet_samples", device="cpu"
):
    """
    Test the ASR model with the generated Vietnamese samples.

    Args:
        model_id: The model ID or path
        samples_dir: Directory containing the test samples
        device: Device to run inference on (cpu or cuda)
    """
    print(f"Testing model {model_id} with Vietnamese samples from {samples_dir}")

    # Make sure checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)

    # Load metadata
    metadata_path = os.path.join(samples_dir, "metadata.json")
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # Create the ASR model
    print("Loading ASR model...")
    model = create_asr_model(model_id, device)

    # Test on each sample
    results = []
    for sample in metadata:
        sample_id = sample["id"]
        audio_path = sample["audio_path"]
        expected_text = sample["transcription"]

        print(f"\nTesting sample {sample_id}: {audio_path}")
        print(f"Expected: {expected_text}")

        # Run inference
        result = model.transcribe(audio_path)
        predicted_text = result["text"]

        print(f"Predicted: {predicted_text}")
        print(f"Duration: {result['duration']:.2f}s")

        # Calculate simple character error rate
        if len(expected_text) > 0:
            cer = 1.0  # Default to 1.0 (100% error) for empty predictions
            if len(predicted_text) > 0:
                # Simple character error rate calculation
                min_len = min(len(expected_text), len(predicted_text))
                matches = sum(1 for i in range(min_len) if expected_text[i] == predicted_text[i])
                cer = 1.0 - (matches / max(len(expected_text), len(predicted_text)))
            print(f"Character Error Rate: {cer:.2%}")
        else:
            cer = 0.0  # If expected is empty, no error
            print("Cannot calculate CER (expected text is empty)")

        results.append(
            {
                "sample_id": sample_id,
                "expected": expected_text,
                "predicted": predicted_text,
                "cer": cer,
            }
        )

    # Print summary
    print("\n--- Testing Summary ---")
    print(f"Model: {model_id}")
    print(f"Samples: {len(results)}")

    if results:
        avg_cer = sum(r["cer"] for r in results) / len(results)
        print(f"Average Character Error Rate: {avg_cer:.2%}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ASR model with Vietnamese samples")
    parser.add_argument(
        "--model", default="tuandunghcmut/PhoWhisper-tiny-CTC", help="Model ID or path"
    )
    parser.add_argument(
        "--samples", default="examples/viet_samples", help="Directory with test samples"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")

    args = parser.parse_args()
    test_model_with_samples(args.model, args.samples, args.device)
