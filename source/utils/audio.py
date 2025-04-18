import numpy as np
import librosa
import soundfile as sf
import torch
import logging
from pathlib import Path
from typing import Tuple, Union, Optional

logger = logging.getLogger(__name__)

def load_audio(
    file_path: Union[str, Path], 
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True,
    max_duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file with proper resampling.
    
    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate
        mono: Whether to convert to mono
        normalize: Whether to normalize the audio
        max_duration: Maximum duration in seconds (None for no limit)
        
    Returns:
        Tuple of (audio array, sample rate)
    """
    try:
        # Load audio file
        if max_duration is not None:
            # Calculate duration in samples
            duration_samples = int(max_duration * target_sr)
            # Load with duration limit
            audio, sr = librosa.load(
                file_path, 
                sr=target_sr, 
                mono=mono, 
                duration=max_duration
            )
        else:
            audio, sr = librosa.load(
                file_path, 
                sr=target_sr, 
                mono=mono
            )
        
        # Normalize if requested
        if normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio, sr
    
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise

def save_audio(
    audio: np.ndarray, 
    file_path: Union[str, Path], 
    sample_rate: int = 16000
) -> None:
    """
    Save audio to a file.
    
    Args:
        audio: Audio array
        file_path: Path to save the audio file
        sample_rate: Sample rate
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save audio
        sf.write(file_path, audio, sample_rate)
    except Exception as e:
        logger.error(f"Error saving audio file {file_path}: {e}")
        raise

def get_audio_duration(file_path: Union[str, Path]) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    try:
        return librosa.get_duration(filename=file_path)
    except Exception as e:
        logger.error(f"Error getting duration of audio file {file_path}: {e}")
        raise

def audio_to_features(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    top_db: int = 80
) -> np.ndarray:
    """
    Convert audio to mel spectrogram features.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length
        top_db: Top dB
        
    Returns:
        Mel spectrogram features
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convert to log scale
    log_mel = librosa.power_to_db(mel_spec, top_db=top_db)
    
    # Normalize
    log_mel = (log_mel + top_db) / top_db
    
    return log_mel

def extract_features_for_model(
    audio: np.ndarray, 
    processor,
    sample_rate: int = 16000
) -> torch.Tensor:
    """
    Extract features for the WhisperCTC model.
    
    Args:
        audio: Audio array
        processor: Whisper processor
        sample_rate: Sample rate
        
    Returns:
        Input features tensor
    """
    features = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_features
    
    return features 