import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import hydra
import librosa
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from transformers import AutoProcessor

from src.models.ctc_model import WhisperCTCModel

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model loading and inference for Vietnamese speech recognition with CTC.
    Supports both CPU and GPU inference.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.device = torch.device(config.inference.device)
        self.half_precision = config.inference.half_precision

        # Initialize model and processor
        self._load_processor()
        self._load_model()

    def _load_processor(self) -> None:
        """Load the processor from pretrained model."""
        logger.info(f"Loading processor from {self.config.model.name}")
        self.processor = AutoProcessor.from_pretrained(self.config.model.name)

    def _load_model(self) -> None:
        """Load the model from checkpoint or huggingface hub."""
        checkpoint_path = self.config.model.checkpoint_path

        # Check if checkpoint exists locally
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading model from local checkpoint: {checkpoint_path}")
            self.model = WhisperCTCModel.load_from_checkpoint(checkpoint_path)
        else:
            # Try downloading from HuggingFace Hub
            try:
                logger.info(
                    f"Downloading model from HuggingFace: {self.config.model.huggingface_repo_id}"
                )
                downloaded_path = hf_hub_download(
                    repo_id=self.config.model.huggingface_repo_id,
                    filename=self.config.model.huggingface_filename,
                    local_dir="./temp_checkpoints",
                )
                self.model = WhisperCTCModel.load_from_checkpoint(downloaded_path)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise ValueError(f"Could not load model from checkpoint or HuggingFace: {e}")

        # Move model to device and set evaluation mode
        self.model.to(self.device)
        self.model.eval()

        # Set half precision if configured (generally not recommended for CPU)
        if self.half_precision and self.device.type == "cuda":
            logger.info("Using half precision (only used for CUDA devices)")
            self.model.half()

        logger.info(f"Model loaded successfully on {self.device}")

    def transcribe_file(self, audio_path: str | Path) -> str:
        """
        Transcribe audio from a file path.

        Args:
            audio_path: Path to the audio file

        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.config.audio.sampling_rate)
        return self.transcribe_audio(audio)

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """
        Transcribe audio from a numpy array.

        Args:
            audio: Audio array (samples)

        Returns:
            Transcribed text
        """
        # Prepare input features
        with torch.no_grad():
            input_features = self.processor(
                audio,
                sampling_rate=self.config.audio.sampling_rate,
                return_tensors="pt",
            ).input_features.to(self.device)

            # Get model predictions
            if self.half_precision and self.device.type == "cuda":
                input_features = input_features.half()

            outputs = self.model(input_features=input_features)

            # Decode predictions
            predicted_text = self._decode_predictions(outputs.logits)

            return predicted_text

    def _decode_predictions(self, logits: torch.Tensor) -> str:
        """
        Decode CTC logits to text.

        Args:
            logits: Model logits (time, batch, vocab)

        Returns:
            Decoded text
        """
        logits = logits.detach().cpu()

        # Get class indices
        logits = logits.transpose(0, 1)  # (batch, time, vocab)
        class_indices = torch.argmax(logits, dim=2)[0]  # Take first batch item

        # Remove blanks (pad tokens)
        seq_no_blank = class_indices[class_indices != self.processor.tokenizer.pad_token_id]

        # Collapse repeats
        seq_collapsed = []
        prev_token = -1
        for token in seq_no_blank:
            if token != prev_token:
                seq_collapsed.append(token.item())
                prev_token = token

        # Decode to text
        text = self.processor.decode(seq_collapsed, skip_special_tokens=True)

        return text

    def batch_transcribe_files(self, audio_paths: list[str | Path]) -> list[str]:
        """
        Transcribe multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of transcribed texts
        """
        return [self.transcribe_file(path) for path in audio_paths]
