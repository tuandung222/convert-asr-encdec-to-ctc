import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.serialization
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download
from torch import nn
from transformers import AutoConfig, AutoProcessor, WhisperFeatureExtractor, WhisperProcessor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("asr-model")

# Add necessary classes to safe globals for checkpoint loading
# torch.serialization.add_safe_globals([
#     WhisperProcessor,
#     WhisperFeatureExtractor,
#     np.core.multiarray._reconstruct,  # Add numpy reconstructors to safe globals
#     np.ndarray,
#     np.dtype
# ])


class PhoWhisperCTCModel(nn.Module):
    """
    PhoWhisperCTCModel that uses a WhisperEncoder and adds a CTC head on top.
    This matches the architecture from the notebook implementation.
    """

    def __init__(self, encoder, dim, vocab_size):
        super().__init__()
        self.encoder = encoder
        # Use Sequential to match the structure used during training
        self.ctc_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, vocab_size)
        )

    def forward(self, input_features, attention_mask=None):
        # Get encoder output
        encoder_out = self.encoder(input_features, attention_mask=attention_mask).last_hidden_state

        # Apply CTC head to get logits
        logits = self.ctc_head(encoder_out)

        return logits

    def debug_output(self, logits, tokenizer, top_n=5):
        """Debug the model's output by showing top predictions at each timestamp."""
        # Get predictions for each timestamp
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get top N predictions for each timestamp
        values, indices = torch.topk(probs, top_n, dim=-1)

        # Convert to numpy for easier handling
        values = values.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()

        results = []

        # For each timestamp
        for t in range(indices.shape[1]):
            tokens = []
            for i in range(top_n):
                token_id = int(indices[0, t, i])  # Convert to Python int
                prob = float(values[0, t, i])  # Convert to Python float
                if token_id == tokenizer.pad_token_id:
                    token_text = "[PAD]"
                else:
                    token_text = tokenizer.convert_ids_to_tokens(token_id)
                tokens.append((token_id, token_text, prob))
            results.append(tokens)

        return results


class ASRInferenceModel:
    """
    ASR inference model wrapper for CTC-based Vietnamese speech recognition.
    This class handles model loading, prediction, and processing for both
    CTC-based and encoder-decoder models.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        """
        Initialize the ASR model.

        Args:
            model_id: Model identifier on Hugging Face Hub
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")

        # Initialize model and processor
        self.processor = None
        self.model = None
        self.vocab_size = 0

        # Load the model and processor
        self._download_checkpoint()
        self._load_model()

        logger.info(f"ASR model initialized on {self.device}")

    def _download_checkpoint(self) -> str:
        """
        Download checkpoint from Hugging Face Hub.

        Returns:
            Path to the downloaded checkpoint
        """
        try:
            logger.info(f"Downloading checkpoint from {self.model_id}")

            # Check if model_id is a local path
            if os.path.exists(self.model_id):
                logger.info(f"Using local model path: {self.model_id}")
                return self.model_id

            # Try to find in checkpoints directory first
            checkpoint_path = os.path.join("checkpoints", "best-val_wer=0.3986.ckpt")
            if os.path.exists(checkpoint_path):
                logger.info(f"Found checkpoint at {checkpoint_path}")
                return checkpoint_path

            # Download checkpoint file directly
            try:
                checkpoint_path = hf_hub_download(
                    repo_id=self.model_id,
                    filename="best-val_wer=0.3986.ckpt",  # This is the known checkpoint filename
                    local_dir="./checkpoints",
                )

                logger.info(f"Checkpoint downloaded to {checkpoint_path}")
                return checkpoint_path
            except Exception as e:
                logger.warning(f"Could not download checkpoint directly: {e}")

                # Try downloading the whole repository
                try:
                    repo_path = snapshot_download(self.model_id)
                    logger.info(f"Repository downloaded to {repo_path}")

                    # Look for checkpoint files
                    ckpt_files = [
                        os.path.join(repo_path, f)
                        for f in os.listdir(repo_path)
                        if f.endswith(".ckpt")
                    ]
                    if ckpt_files:
                        logger.info(f"Found checkpoint: {ckpt_files[0]}")
                        return ckpt_files[0]
                    else:
                        raise ValueError(f"No checkpoint files found in repository {self.model_id}")
                except Exception as repo_error:
                    raise ValueError(
                        f"Failed to download checkpoint or repository: {str(e)} -> {str(repo_error)}"
                    )
        except Exception as e:
            logger.error(f"Error downloading checkpoint: {e}")
            raise

    def _load_model(self) -> None:
        """
        Load model and processor from the downloaded checkpoint.
        This handles both CTC and encoder-decoder models.
        """
        try:
            # Load processor from the base model instead of fine-tuned model
            logger.info("Loading processor from base model vinai/PhoWhisper-tiny...")
            self.processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-tiny")

            # Check if we have a Lightning checkpoint (.ckpt file)
            # Look for .ckpt file in the downloaded directory
            if os.path.isdir(self.model_id):
                ckpt_files = [f for f in os.listdir(self.model_id) if f.endswith(".ckpt")]
                if ckpt_files:
                    ckpt_path = os.path.join(self.model_id, ckpt_files[0])
                else:
                    ckpt_path = None
            elif os.path.isfile(self.model_id) and self.model_id.endswith(".ckpt"):
                # The model_id itself is a checkpoint file
                ckpt_path = self.model_id
            else:
                # Try to find checkpoint file in the expected path
                expected_path = os.path.join("./checkpoints", "best-val_wer=0.3986.ckpt")
                if os.path.exists(expected_path):
                    ckpt_path = expected_path
                else:
                    ckpt_path = None

            if ckpt_path:
                # We have a Lightning checkpoint file
                logger.info(f"Found Lightning checkpoint: {ckpt_path}")

                # Load checkpoint with weights_only=False to handle the PyTorch 2.6+ behavior
                try:
                    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                except TypeError:
                    # For older PyTorch versions that don't have weights_only parameter
                    checkpoint = torch.load(ckpt_path, map_location="cpu")

                # Get configuration
                config = AutoConfig.from_pretrained("vinai/PhoWhisper-tiny")

                # Create encoder
                self.encoder = WhisperEncoder(config=config)

                # Create CTC head with fixed vocabulary size from checkpoint
                # Using 50258 as the fixed vocab size from the checkpoint
                fixed_vocab_size = 50258  # The vocab size used during training
                self.model = PhoWhisperCTCModel(
                    encoder=self.encoder, dim=config.d_model, vocab_size=fixed_vocab_size
                )

                # Load state dict from checkpoint
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    # Map keys from Lightning checkpoint to our model
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith("encoder."):
                            new_state_dict[k] = v
                        elif k.startswith("ctc_head."):
                            # Check if it's from the sequential model
                            if k.startswith("ctc_head.layers."):
                                # Match the exact layer indexes in Sequential
                                if "layers.0" in k:  # First linear layer
                                    new_key = k.replace("ctc_head.layers.0", "ctc_head.0")
                                    new_state_dict[new_key] = v
                                elif "layers.2" in k:  # LayerNorm
                                    new_key = k.replace("ctc_head.layers.2", "ctc_head.2")
                                    new_state_dict[new_key] = v
                                elif "layers.3" in k:  # Second linear layer
                                    new_key = k.replace("ctc_head.layers.3", "ctc_head.3")
                                    new_state_dict[new_key] = v
                                else:
                                    # Keep other layers unchanged
                                    new_state_dict[k.replace("ctc_head.layers.", "ctc_head.")] = v
                            else:
                                # No "layers" prefix, just use as is
                                new_state_dict[k] = v

                    # Load the state dict
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        new_state_dict, strict=False
                    )

                    if missing_keys:
                        logger.warning(f"Missing keys when loading state dict: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(
                            f"Unexpected keys when loading state dict: {unexpected_keys}"
                        )
                else:
                    logger.error("No state_dict found in checkpoint")
                    raise ValueError("Invalid checkpoint format")
            else:
                # Fall back to loading from standard Hugging Face model
                logger.info("No Lightning checkpoint found, using standard model loading")

                # Load base encoder from pre-trained checkpoint
                from transformers import WhisperModel

                whisper_model = WhisperModel.from_pretrained("vinai/PhoWhisper-tiny")
                encoder = whisper_model.encoder

                # Create CTC model
                self.vocab_size = len(self.processor.tokenizer)
                logger.info(f"Vocabulary size: {self.vocab_size}")

                self.model = PhoWhisperCTCModel(
                    encoder=encoder, dim=encoder.config.d_model, vocab_size=self.vocab_size
                )

                # Try to load CTC head weights if they exist
                model_bin_path = os.path.join(self.model_id, "pytorch_model.bin")
                if os.path.exists(model_bin_path):
                    state_dict = torch.load(model_bin_path, map_location="cpu")

                    # Filter state dict for just the parts we need
                    model_state_dict = {}
                    for k, v in state_dict.items():
                        # Keep encoder weights and CTC head
                        if k.startswith("encoder.") or k.startswith("ctc_head."):
                            model_state_dict[k] = v

                    # Load state dict into model
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        model_state_dict, strict=False
                    )

                    if missing_keys:
                        logger.warning(f"Missing keys when loading state dict: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(
                            f"Unexpected keys when loading state dict: {unexpected_keys}"
                        )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dict with transcription result and metadata
        """
        try:
            # Load and preprocess audio
            logger.info(f"Processing audio file: {audio_path}")

            # Load audio using torchaudio
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                # Calculate duration
                duration = waveform.shape[1] / sample_rate
            except Exception as e:
                logger.warning(f"Error loading audio with torchaudio: {e}. Trying librosa...")
                # Fallback to librosa
                import librosa

                waveform, sample_rate = librosa.load(audio_path, sr=16000)
                # Convert to torch tensor
                waveform = torch.from_numpy(waveform).unsqueeze(0)
                duration = len(waveform[0]) / sample_rate

            # Resample if needed
            if sample_rate != 16000:
                transform = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = transform(waveform)
                sample_rate = 16000

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Make sure duration is valid
            duration = max(0.1, duration)  # At least 0.1 seconds

            # Extract features
            inputs = self.processor.feature_extractor(
                waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt"
            )

            # Move inputs to device
            input_features = inputs.input_features.to(self.device)

            # Generate logits
            with torch.no_grad():
                logits = self.model(input_features)

            # Get debug information (top predictions)
            debug_info = None
            if hasattr(self.model, "debug_output"):
                debug_info = self.model.debug_output(logits, self.processor.tokenizer, top_n=3)

                # Print some debug info
                logger.info("Top predictions for first 5 timestamps:")
                for t in range(min(5, len(debug_info))):
                    logger.info(f"Timestamp {t}:")
                    for token_id, token_text, prob in debug_info[t]:
                        logger.info(f"  {token_id} ({token_text}): {prob:.4f}")

            # Decode predictions with a simpler approach - first get the most likely token at each timestamp
            predicted_ids = torch.argmax(logits, dim=-1)[0]  # Take first batch item

            # Convert to list
            predicted_ids = predicted_ids.cpu().tolist()

            # Remove pad tokens (CTC blank)
            predicted_ids_filtered = [
                id for id in predicted_ids if id != self.processor.tokenizer.pad_token_id
            ]

            # Simple CTC decoding: collapse repeated tokens
            collapsed_ids = []
            for i, id in enumerate(predicted_ids_filtered):
                if i == 0 or id != predicted_ids_filtered[i - 1]:
                    collapsed_ids.append(id)

            # Decode to text
            transcription = self.processor.tokenizer.decode(collapsed_ids, skip_special_tokens=True)

            # Return results
            return {"text": transcription, "duration": duration, "debug_info": debug_info}

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {
                "text": f"Error: {str(e)}",
                "duration": 1.0,  # Default duration to avoid division by zero
                "error": True,
            }


class ONNXASRInferenceModel(ASRInferenceModel):
    """
    ONNX version of the ASR model for faster inference.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        """
        Initialize the ONNX ASR inference model.

        Args:
            model_id: Path to the model or model identifier
            device: Device to run inference on
        """
        self.onnx_path = None
        self.ort_session = None
        super().__init__(model_id, device)

    def _find_onnx_files(self, directory):
        """
        Find ONNX model files in the given directory

        Args:
            directory: Directory to search in

        Returns:
            Dict with paths to model files
        """
        logger.info(f"Looking for ONNX files in {directory}")
        files = {}

        # First check for int8 quantized model
        int8_model_path = os.path.join(directory, "model_int8.onnx")
        if os.path.exists(int8_model_path):
            files["model"] = int8_model_path
            logger.info(f"Found INT8 quantized ONNX model: {int8_model_path}")
            return files

        # Check for the main model file
        model_path = os.path.join(directory, "model.onnx")
        if os.path.exists(model_path):
            files["model"] = model_path
            logger.info(f"Found ONNX model: {model_path}")

        # If directory contains multiple .onnx files, use the first one
        if not files:
            onnx_files = [f for f in os.listdir(directory) if f.endswith(".onnx")]
            if onnx_files:
                model_path = os.path.join(directory, onnx_files[0])
                files["model"] = model_path
                logger.info(f"Found ONNX model: {model_path}")

        return files

    def _load_model(self) -> None:
        """
        Load the ONNX model
        """
        try:
            # Always load the processor from the base model first
            logger.info("Loading processor from base model vinai/PhoWhisper-tiny...")
            self.processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-tiny")

            # Check if ONNX model files exist in the specified directory
            model_files = {}
            if os.path.isdir(self.model_id):
                model_files = self._find_onnx_files(self.model_id)

            # If no ONNX files found, try the default location
            if not model_files:
                default_path = os.path.join("models", "onnx", os.path.basename(self.model_id))
                if os.path.isdir(default_path):
                    model_files = self._find_onnx_files(default_path)
                    if model_files:
                        logger.info(f"Found ONNX files in default path: {default_path}")

            # If no ONNX files found, attempt to convert from PyTorch
            if not model_files:
                logger.info(f"No ONNX files found. Attempting to convert PyTorch model to ONNX...")
                model_files = self._convert_to_onnx()

            # If we have ONNX files, load them
            if model_files:
                self._load_onnx_session(model_files)
                logger.info(f"ONNX model loaded successfully on {self.device}")
                return

            # If we reach here, ONNX loading failed or no files were found
            raise FileNotFoundError("No ONNX model files found and conversion failed")

        except Exception as e:
            logger.warning(f"Error loading ONNX model: {e}")
            import traceback

            logger.warning(traceback.format_exc())

            # Fall back to standard PyTorch model
            logger.info("Falling back to standard PyTorch model...")

            # Make sure to preserve the processor we loaded from the base model
            processor = self.processor

            # Load standard model
            super()._load_model()

            # Ensure we're using the processor from the base model
            if processor is not None:
                logger.info("Using pre-loaded processor from base model")
                self.processor = processor

    def _load_onnx_session(self, model_files):
        """
        Load the ONNX model using ONNX Runtime

        Args:
            model_files: Dict with paths to model files
        """
        try:
            import onnxruntime as ort

            # Set execution providers
            execution_providers = ["CPUExecutionProvider"]
            if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                execution_providers.insert(0, "CUDAExecutionProvider")
                logger.info("Using CUDA for ONNX Runtime")

            # Create session options
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Load model
            model_path = model_files.get("model")
            if not model_path:
                raise ValueError("No model file found in model_files dict")

            logger.info(f"Loading ONNX model from {model_path}")

            try:
                self.ort_session = ort.InferenceSession(
                    model_path, providers=execution_providers, sess_options=session_options
                )
                # Store path for later reference
                self.onnx_path = model_path

                logger.info(
                    f"ONNX model loaded successfully with providers: {self.ort_session.get_providers()}"
                )

                # Log model inputs and outputs
                inputs = self.ort_session.get_inputs()
                outputs = self.ort_session.get_outputs()

                logger.info(f"Model inputs: {[i.name for i in inputs]}")
                logger.info(f"Model outputs: {[o.name for o in outputs]}")

            except Exception as e:
                logger.error(f"Error loading ONNX model: {e}")
                logger.warning(f"Error loading ONNX model: {e}")

                # If this is a quantized model that failed to load, try the regular model
                if "_int8" in model_path:
                    logger.warning("Quantized model failed to load. Falling back to FP32 model.")
                    # Remove the failed INT8 model
                    try:
                        os.remove(model_path)
                        logger.info(f"Removed problematic quantized model: {model_path}")
                    except Exception as rm_err:
                        logger.warning(f"Could not remove quantized model: {rm_err}")

                    # Try to load the regular FP32 model
                    fp32_path = model_path.replace("_int8", "")
                    if os.path.exists(fp32_path):
                        logger.info(f"Trying to load FP32 model: {fp32_path}")
                        self.ort_session = ort.InferenceSession(
                            fp32_path, providers=execution_providers, sess_options=session_options
                        )
                        self.onnx_path = fp32_path
                        logger.info(
                            f"FP32 model loaded successfully with providers: {self.ort_session.get_providers()}"
                        )
                    else:
                        raise ValueError(f"FP32 model not found at {fp32_path}")
                else:
                    raise

        except ImportError:
            logger.warning(
                "ONNX Runtime not installed. Please install with 'pip install onnxruntime'"
            )
            raise
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            raise

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        """
        Transcribe using ONNX model if available, otherwise use PyTorch.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with transcription result and metadata
        """
        # Check if ONNX is available and if the model is an ONNX InferenceSession
        if self.onnx_path is None:
            logger.info("ONNX Runtime not available. Using PyTorch implementation.")
            return super().transcribe(audio_path)

        # Check if self.ort_session is the right type - import onnxruntime first
        import onnxruntime as ort

        if not isinstance(self.ort_session, ort.InferenceSession):
            logger.info("Model is not an ONNX InferenceSession. Using PyTorch implementation.")
            return super().transcribe(audio_path)

        try:
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if needed
            if sample_rate != 16000:
                transform = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = transform(waveform)
                sample_rate = 16000

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Calculate duration
            duration = waveform.shape[1] / sample_rate

            # Extract features
            inputs = self.processor.feature_extractor(
                waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="np"
            )

            # Run inference with ONNX
            ort_inputs = {self.ort_session.get_inputs()[0].name: inputs.input_features}

            # Get logits from ONNX model
            logits = self.ort_session.run(None, ort_inputs)[0]

            # Optimized CTC decoding using NumPy for better performance
            import numpy as np

            # 1. Get the most likely token at each timestamp (argmax along vocab dimension)
            predicted_ids = np.argmax(logits[0], axis=-1)

            # 2. Get mask for non-blank tokens (much faster than list comprehension)
            pad_token_id = self.processor.tokenizer.pad_token_id
            non_blank_mask = predicted_ids != pad_token_id
            filtered_ids = predicted_ids[non_blank_mask]

            # 3. Find where consecutive tokens differ (for collapsing repeats)
            if len(filtered_ids) > 0:
                # Append a sentinel value to detect the last change
                padded_ids = np.append(filtered_ids, -1)
                # Find where consecutive values are different (including the sentinel)
                changes = np.where(padded_ids[1:] != padded_ids[:-1])[0]
                # Get the values at change indices (much faster than Python loop)
                collapsed_ids = filtered_ids[changes]
            else:
                collapsed_ids = filtered_ids

            # 4. Convert numpy array to list for the tokenizer
            token_ids = collapsed_ids.tolist()

            # 5. Decode to text
            transcription = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)

            # 6. Remove the first two characters (tokenization bug fix)
            if len(transcription) > 2:
                transcription = transcription[2:]

            # Return results
            return {
                "text": transcription,
                "duration": duration,
            }

        except Exception as e:
            logger.error(f"Error transcribing audio with ONNX: {e}")
            import traceback

            logger.error(traceback.format_exc())
            # Fall back to PyTorch
            return super().transcribe(audio_path)

    def _convert_to_onnx(self):
        """
        Convert the PyTorch model to ONNX format

        Returns:
            Dict with paths to model files
        """
        try:
            import torch

            # Set output directory
            onnx_dir = os.path.join("checkpoints", "onnx")
            os.makedirs(onnx_dir, exist_ok=True)

            # Check if INT8 model already exists
            int8_onnx_path = os.path.join(onnx_dir, "model_int8.onnx")
            if os.path.exists(int8_onnx_path):
                logger.info(
                    f"INT8 ONNX model already exists at {int8_onnx_path}. Using existing model."
                )
                return {"model": int8_onnx_path}

            # Check if FP32 model already exists
            onnx_path = os.path.join(onnx_dir, "model.onnx")
            if os.path.exists(onnx_path):
                logger.info(f"FP32 ONNX model already exists at {onnx_path}. Using existing model.")

                # Try to quantize the existing model if onnxruntime is available
                try:
                    import onnxruntime as ort
                    from onnxruntime.quantization import QuantType, quantize_dynamic

                    logger.info(f"Applying INT8 quantization to existing ONNX model...")
                    quantize_dynamic(
                        model_input=onnx_path,
                        model_output=int8_onnx_path,
                        per_channel=False,
                        weight_type=QuantType.QUInt8,
                        op_types_to_quantize=["MatMul"],
                    )
                    logger.info(f"INT8 quantization completed: {int8_onnx_path}")

                    # Use quantized model if available
                    if os.path.exists(int8_onnx_path):
                        return {"model": int8_onnx_path}

                except ImportError:
                    logger.warning(
                        "ONNX Runtime quantization tools not available. Using FP32 model."
                    )
                except Exception as quant_error:
                    logger.warning(f"INT8 quantization failed: {quant_error}. Using FP32 model.")

                return {"model": onnx_path}

            # If no existing models, proceed with conversion
            logger.info(f"Converting PyTorch model to ONNX...")

            # First, load the PyTorch model (standard model)
            pytorch_model = ASRInferenceModel(self.model_id, self.device)

            # Ensure the model is loaded
            if pytorch_model.model is None:
                raise ValueError("Failed to load PyTorch model for conversion")

            # Create dummy input for tracing
            dummy_input = torch.randn(1, 80, 3000).to(self.device)

            # Set the model to eval mode
            pytorch_model.model.eval()

            # Export the model to ONNX
            logger.info(f"Exporting model to ONNX at {onnx_path}")

            with torch.no_grad():
                torch.onnx.export(
                    pytorch_model.model,
                    dummy_input,
                    onnx_path,
                    input_names=["input_features"],
                    output_names=["logits"],
                    dynamic_axes={
                        "input_features": {0: "batch_size", 2: "sequence_length"},
                        "logits": {0: "batch_size", 1: "sequence_length"},
                    },
                    opset_version=14,
                    verbose=False,
                )

            logger.info(f"Model exported to ONNX successfully: {onnx_path}")

            # Verify the exported model
            try:
                import onnx

                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX model verification passed")

                # Apply INT8 quantization if onnxruntime is available
                try:
                    import onnxruntime as ort
                    from onnxruntime.quantization import QuantType, quantize_dynamic

                    logger.info(f"Applying INT8 quantization to ONNX model...")
                    quantize_dynamic(
                        model_input=onnx_path,
                        model_output=int8_onnx_path,
                        per_channel=False,
                        weight_type=QuantType.QUInt8,  # Use unsigned int8 instead of signed
                        op_types_to_quantize=[
                            "MatMul"
                        ],  # Only quantize MatMul operations, not Conv
                    )
                    logger.info(f"INT8 quantization completed: {int8_onnx_path}")

                    # Use quantized model if available
                    if os.path.exists(int8_onnx_path):
                        return {"model": int8_onnx_path}

                except ImportError:
                    logger.warning(
                        "ONNX Runtime quantization tools not available. Using FP32 model."
                    )
                except Exception as quant_error:
                    logger.warning(f"INT8 quantization failed: {quant_error}. Using FP32 model.")

            except ImportError:
                logger.warning("ONNX package not installed. Skipping model verification.")
            except Exception as verify_error:
                logger.warning(f"ONNX model verification failed: {verify_error}")

            # Return the model files (original FP32 if quantization failed)
            return {"model": onnx_path}

        except Exception as e:
            logger.error(f"Error converting model to ONNX: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {}


# class LightningCheckpointInferenceModel(ASRInferenceModel):
#     """
#     Inference model that directly loads a PyTorch Lightning checkpoint.
#     This matches exactly the approach used in the evaluation notebook.
#     """

#     def __init__(
#         self,
#         model_id: str,
#         device: str = "cpu"
#     ):
#         """
#         Initialize the LightningCheckpoint ASR model.

#         Args:
#             model_id: Path to the checkpoint or model identifier
#             device: Device to run inference on
#         """
#         super().__init__(model_id, device)

#     def _load_model(self) -> None:
#         """
#         Load model using PyTorch Lightning's load_from_checkpoint
#         """
#         try:
#             # Import the Lightning module class - the same one used during training
#             import sys
#             sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#             from notebooks.evalutation_after_training import PhoWhisperLightningModule

#             # Download or find the checkpoint
#             checkpoint_path = self._download_checkpoint()

#             # Always load the processor from the base model first
#             logger.info("Loading processor from base model vinai/PhoWhisper-tiny...")
#             self.processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-tiny")

#             # Load using the Lightning API
#             logger.info(f"Loading model from Lightning checkpoint: {checkpoint_path}")

#             # Load checkpoint without the processor to avoid the preprocessor_config.json error
#             # We'll explicitly set the processor after loading
#             lightning_module = PhoWhisperLightningModule.load_from_checkpoint(
#                 checkpoint_path,
#                 map_location=self.device
#             )

#             # Set our processor from base model
#             lightning_module.processor = self.processor

#             # Move to device and set to eval mode
#             lightning_module.to(self.device)
#             lightning_module.eval()

#             # Create a thin wrapper for compatibility with our inference API
#             self.lightning_model = lightning_module
#             self.model = self.lightning_model

#             logger.info(f"Model loaded successfully on {self.device}")

#         except Exception as e:
#             logger.error(f"Error loading Lightning model: {e}")
#             import traceback
#             logger.error(traceback.format_exc())

#             # Fall back to standard loading
#             logger.info("Falling back to standard model loading...")
#             super()._load_model()

#     def transcribe(self, audio_path: str) -> Dict[str, Any]:
#         """
#         Transcribe an audio file using the Lightning model.

#         Args:
#             audio_path: Path to audio file

#         Returns:
#             Dict with transcription result and metadata
#         """
#         try:
#             # Load and preprocess audio in same way as the evaluation script
#             import librosa

#             # Load audio file
#             logger.info(f"Processing audio file: {audio_path}")
#             audio, sample_rate = librosa.load(audio_path, sr=16000)

#             # Use the processor to get input features
#             input_features = self.processor(
#                 audio,
#                 sampling_rate=16000,
#                 return_tensors="pt"
#             ).input_features.to(self.device)

#             # Generate logits using the model's forward function
#             with torch.no_grad():
#                 # The model returns logits directly, not an object with .logits attribute
#                 logits = self.model(input_features=input_features)

#             # Decode the logits using the model's built-in method
#             predicted_text = self.model.ctc_decode(logits, self.processor)[0]

#             # Following the notebook: remove the first two characters (due to tokenization mistake)
#             predicted_text = predicted_text[2:]

#             # Calculate duration
#             duration = len(audio) / 16000

#             # Return results
#             return {
#                 "text": predicted_text,
#                 "duration": duration
#             }

#         except Exception as e:
#             logger.error(f"Error transcribing with Lightning model: {e}")
#             import traceback
#             logger.error(traceback.format_exc())

#             # Fall back to standard transcription
#             return super().transcribe(audio_path)


def create_asr_model(
    model_id: str, device: str = "cpu", model_type: str = "onnx"
) -> ASRInferenceModel:
    """
    Factory function to create an ASR model.

    Args:
        model_id: Model identifier or path
        device: Device to run inference on ('cpu' or 'cuda')
        model_type: Model type ('pytorch', 'onnx', or 'lightning', default: 'lightning')

    Returns:
        ASR inference model
    """
    # if model_type.lower() == "lightning":
    #     return LightningCheckpointInferenceModel(model_id, device)

    if model_type.lower() == "onnx":
        # NOTE: This is the optimized ONNX model for faster inference
        from .improved_inference_model import OptimizedONNXASRInferenceModel

        return OptimizedONNXASRInferenceModel(model_id, device)
        # return ONNXASRInferenceModel(model_id, device)
    elif model_type.lower() == "pytorch":
        return ASRInferenceModel(model_id, device)
    else:
        logger.error(f"Invalid model type: {model_type}")
        raise ValueError(f"Invalid model type: {model_type}")
