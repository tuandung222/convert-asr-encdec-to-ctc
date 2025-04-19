import os
import torch
from datasets import load_dataset
from transformers import AutoProcessor
import soundfile as sf
import numpy as np

def download_vietbud_test_data(num_samples=5, output_dir="test_data"):
    """
    Download a few examples from VietBud500 test dataset for testing.
    
    Args:
        num_samples: Number of test samples to download
        output_dir: Directory to save the test samples
    """
    print(f"Downloading {num_samples} test samples from VietBud500...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # URL for test data
    test_url = "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/test-00000-of-00002-531c1d81edb57297.parquet"
    
    # Load test dataset
    dataset = load_dataset(
        "parquet",
        data_files={"test": test_url},
    )
    
    # Get the test dataset
    test_dataset = dataset["test"]
    
    # Get sampling rate
    sampling_rate = test_dataset.features["audio"].sampling_rate
    print(f"Audio sampling rate: {sampling_rate} Hz")
    
    # Select a subset of examples
    test_subset = test_dataset.select(range(min(num_samples, len(test_dataset))))
    
    # Save each example
    metadata = []
    for i, example in enumerate(test_subset):
        # Get audio and transcription
        audio = example["audio"]["array"]
        transcription = example["transcription"]
        
        # Create file paths
        audio_path = os.path.join(output_dir, f"test_sample_{i+1}.wav")
        transcription_path = os.path.join(output_dir, f"test_sample_{i+1}.txt")
        
        # Save audio as WAV file
        sf.write(audio_path, audio, sampling_rate)
        
        # Save transcription as text file
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(transcription)
        
        metadata.append({
            "id": i+1,
            "audio_path": audio_path,
            "transcription_path": transcription_path,
            "transcription": transcription,
            "duration": len(audio) / sampling_rate
        })
        
        print(f"Saved sample {i+1}: {audio_path}")
        print(f"Transcription: {transcription}")
        print(f"Duration: {len(audio) / sampling_rate:.2f} seconds")
        print("-" * 40)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(f"ID: {item['id']}\n")
            f.write(f"Audio: {item['audio_path']}\n")
            f.write(f"Transcription: {item['transcription']}\n")
            f.write(f"Duration: {item['duration']:.2f} seconds\n")
            f.write("-" * 40 + "\n")
    
    print(f"Downloaded {len(metadata)} test samples to {output_dir}")
    print(f"Metadata saved to {metadata_path}")
    
    return test_subset

if __name__ == "__main__":
    download_vietbud_test_data(num_samples=5, output_dir="examples/vietbud_samples") 