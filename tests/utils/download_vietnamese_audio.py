import os
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf


def download_audio_sample(url, output_path, sample_rate=16000):
    """Download an audio sample and convert it to 16kHz WAV format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Downloading audio from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Save the downloaded file to a temporary path
    temp_path = output_path + ".temp"
    with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded to {temp_path}, converting to WAV format...")

    # Load the audio and convert to target sample rate
    audio, sr = librosa.load(temp_path, sr=sample_rate)

    # Save as WAV
    sf.write(output_path, audio, sample_rate)

    # Clean up the temporary file
    os.remove(temp_path)

    print(f"Saved audio to {output_path}")
    print(f"Duration: {len(audio) / sample_rate:.2f} seconds")

    return audio, sample_rate


if __name__ == "__main__":
    # URL to a Vietnamese speech sample (replace with a working URL)
    url = "https://cdn.forvo.com/mp3/34127418/102/34127418_102_23536.mp3"  # This is "Xin chào" (Hello) in Vietnamese
    output_path = "examples/real_vietnamese_sample.wav"

    download_audio_sample(url, output_path)
