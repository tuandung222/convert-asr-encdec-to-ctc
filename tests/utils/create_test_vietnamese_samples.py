import json
import os

import numpy as np
import soundfile as sf


def create_test_data(output_dir="examples/viet_samples", num_samples=5):
    """
    Create synthetic Vietnamese test data.

    Args:
        output_dir: Directory to save the test samples
        num_samples: Number of test samples to create
    """
    print(f"Creating {num_samples} Vietnamese test samples...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Sample Vietnamese phrases
    vietnamese_phrases = [
        "Xin chào thế giới",
        "Tôi đang học tiếng Việt",
        "Hôm nay thời tiết đẹp quá",
        "Bạn có khỏe không?",
        "Cảm ơn bạn rất nhiều",
        "Tôi thích ăn phở",
        "Việt Nam là một đất nước tuyệt vời",
        "Tôi rất thích học công nghệ",
        "Hà Nội là thủ đô của Việt Nam",
        "Chúc bạn một ngày tốt lành",
    ]

    # Sampling rate (16kHz is standard for speech)
    sample_rate = 16000

    # Generate silent audio for each phrase
    metadata = []
    for i in range(min(num_samples, len(vietnamese_phrases))):
        # Get a phrase
        phrase = vietnamese_phrases[i]

        # Generate a silence of random duration (1-3 seconds)
        duration = np.random.uniform(1.0, 3.0)
        audio = np.zeros(int(sample_rate * duration))

        # Create file paths
        audio_path = os.path.join(output_dir, f"test_sample_{i+1}.wav")
        transcription_path = os.path.join(output_dir, f"test_sample_{i+1}.txt")

        # Save audio as WAV file
        sf.write(audio_path, audio, sample_rate)

        # Save transcription as text file
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(phrase)

        metadata.append(
            {
                "id": i + 1,
                "audio_path": audio_path,
                "transcription_path": transcription_path,
                "transcription": phrase,
                "duration": duration,
            }
        )

        print(f"Created sample {i+1}: {audio_path}")
        print(f"Transcription: {phrase}")
        print(f"Duration: {duration:.2f} seconds")
        print("-" * 40)

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Created {len(metadata)} test samples in {output_dir}")
    print(f"Metadata saved to {metadata_path}")

    return metadata


if __name__ == "__main__":
    create_test_data(output_dir="examples/viet_samples", num_samples=5)
