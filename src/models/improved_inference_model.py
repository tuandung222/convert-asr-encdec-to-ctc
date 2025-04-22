import logging
import os
import time
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import WhisperProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("asr-model")

# Try to import optimized audio processing libraries
try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    logger.warning("soundfile not installed. Using torchaudio for processing.")

try:
    import resampy

    HAS_RESAMPY = True
except ImportError:
    HAS_RESAMPY = False
    logger.warning("resampy not installed. Using torchaudio for resampling.")

try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logger.warning("numba not installed. Using standard CTC decoding.")


class OptimizedONNXASRInferenceModel:
    """
    Optimized ONNX version of the ASR model for faster inference.
    Includes performance improvements for preprocessing, inference, and decoding.
    """

    def __init__(self, model_id: str, device: str = "cpu", num_threads: int = 4):
        """
        Initialize the optimized ONNX ASR inference model.

        Args:
            model_id: Path to the model or model identifier
            device: Device to run inference on ('cpu' or 'cuda')
            num_threads: Number of threads to use for intra-op parallelism
        """
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_threads = num_threads
        self.onnx_path = None
        self.ort_session = None
        self.processor = None

        # Initialize model and processor
        self._download_checkpoint()
        self._load_model()

        # Perform model warmup for faster first inference
        self.initialize_with_warmup()

        logger.info(f"Optimized ONNX ASR model initialized on {self.device}")

    def _download_checkpoint(self) -> str:
        """
        Download checkpoint from Hugging Face Hub with optimized retry logic.

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
            checkpoint_path = os.path.join("checkpoints", "onnx", "model_int8.onnx")
            if os.path.exists(checkpoint_path):
                logger.info(f"Found checkpoint at {checkpoint_path}")
                return checkpoint_path

            # Try quantized model next
            checkpoint_path = os.path.join("checkpoints", "onnx", "model.onnx")
            if os.path.exists(checkpoint_path):
                logger.info(f"Found checkpoint at {checkpoint_path}")
                return checkpoint_path

            # Download with exponential backoff retry
            max_retries = 3
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    # Try downloading the ONNX model directly
                    checkpoint_path = hf_hub_download(
                        repo_id=self.model_id,
                        filename="model_int8.onnx",  # Try int8 first
                        local_dir="./checkpoints/onnx",
                    )
                    logger.info(f"INT8 ONNX model downloaded to {checkpoint_path}")
                    return checkpoint_path
                except Exception as e:
                    logger.warning(f"Failed to download INT8 ONNX model: {e}")

                    try:
                        # Try downloading the regular ONNX model
                        checkpoint_path = hf_hub_download(
                            repo_id=self.model_id,
                            filename="model.onnx",
                            local_dir="./checkpoints/onnx",
                        )
                        logger.info(f"ONNX model downloaded to {checkpoint_path}")
                        return checkpoint_path
                    except Exception as e:
                        logger.warning(f"Could not download ONNX model: {e}")

                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2**attempt)
                            logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            logger.warning(
                                "All download attempts failed, will try snapshot download"
                            )

            # Try downloading the whole repository
            try:
                repo_path = snapshot_download(self.model_id)
                logger.info(f"Repository downloaded to {repo_path}")

                # Look for ONNX model files
                onnx_files = [
                    os.path.join(repo_path, f) for f in os.listdir(repo_path) if f.endswith(".onnx")
                ]

                if onnx_files:
                    logger.info(f"Found ONNX model: {onnx_files[0]}")
                    return onnx_files[0]
                else:
                    # We'll need to convert PyTorch model to ONNX
                    logger.info("No ONNX files found in repository, will convert PyTorch model")
                    return self.model_id
            except Exception as repo_error:
                logger.error(f"Failed to download checkpoint or repository: {str(repo_error)}")
                raise ValueError(f"Failed to download model: {str(repo_error)}")

        except Exception as e:
            logger.error(f"Error downloading checkpoint: {e}")
            raise

    def _load_model(self) -> None:
        """
        Load the optimized ONNX model
        """
        try:
            # Always load the processor from the base model
            logger.info("Loading processor from base model vinai/PhoWhisper-tiny...")
            self.processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-tiny")

            # Check if we have an ONNX model already
            model_files = {}
            if os.path.isdir(self.model_id):
                model_files = self._find_onnx_files(self.model_id)

            # If no ONNX files found in the specified directory, try default locations
            if not model_files:
                onnx_dir = os.path.join("checkpoints", "onnx")
                if os.path.isdir(onnx_dir):
                    model_files = self._find_onnx_files(onnx_dir)

                # Also try looking for model in the root of model_id
                if not model_files and os.path.exists(self.model_id):
                    filename = os.path.basename(self.model_id)
                    parent_dir = os.path.dirname(self.model_id)
                    onnx_dir = os.path.join(parent_dir, "onnx")
                    if os.path.isdir(onnx_dir):
                        model_files = self._find_onnx_files(onnx_dir)

            # If still no ONNX files, try to convert from PyTorch
            if not model_files:
                logger.info("No ONNX files found. Converting PyTorch model to ONNX...")
                model_files = self._convert_to_onnx()

            # If we have ONNX files, load them
            if model_files:
                self._load_onnx_session(model_files)
                logger.info(f"ONNX model loaded successfully on {self.device}")
                return

            # If we reach here, ONNX loading failed
            raise FileNotFoundError("No ONNX model files found and conversion failed")

        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            import traceback

            logger.error(traceback.format_exc())
            raise

    def _find_onnx_files(self, directory: str) -> Dict[str, str]:
        """
        Find ONNX model files in the given directory with improved detection

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
            return files

        # Look for any .onnx files
        if os.path.isdir(directory):
            onnx_files = [f for f in os.listdir(directory) if f.endswith(".onnx")]
            if onnx_files:
                # Sort by size (prefer smaller quantized models)
                onnx_files.sort(key=lambda f: os.path.getsize(os.path.join(directory, f)))
                model_path = os.path.join(directory, onnx_files[0])
                files["model"] = model_path
                logger.info(f"Found ONNX model: {model_path}")
                return files

        # Look recursively in subdirectories
        if os.path.isdir(directory):
            for subdir in os.listdir(directory):
                subdir_path = os.path.join(directory, subdir)
                if os.path.isdir(subdir_path):
                    subdir_files = self._find_onnx_files(subdir_path)
                    if subdir_files:
                        return subdir_files

        return files

    def _load_onnx_session(self, model_files: Dict[str, str]) -> None:
        """
        Load the ONNX model using ONNX Runtime with optimized settings

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

            # Create optimized session options
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = self.num_threads
            session_options.inter_op_num_threads = max(1, self.num_threads // 2)
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True

            # Set execution mode
            if hasattr(ort, "ExecutionMode"):
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            # Enable optimization for multi-core CPUs
            if hasattr(session_options, "add_session_config_entry"):
                session_options.add_session_config_entry("session.dynamic_block_base_size", "4")
                session_options.add_session_config_entry("session.force_spinning_thread", "1")

            # Load model
            model_path = model_files.get("model")
            if not model_path:
                raise ValueError("No model file found in model_files dict")

            logger.info(f"Loading ONNX model from {model_path} with optimized settings")

            try:
                self.ort_session = ort.InferenceSession(
                    model_path, providers=execution_providers, sess_options=session_options
                )
                # Store path for later reference
                self.onnx_path = model_path

                logger.info(
                    f"ONNX model loaded successfully with providers: {self.ort_session.get_providers()}"
                )

                # Get and cache input/output names for faster inference
                self.input_name = self.ort_session.get_inputs()[0].name
                self.output_names = [o.name for o in self.ort_session.get_outputs()]

                logger.info(f"Model inputs: {self.input_name}")
                logger.info(f"Model outputs: {self.output_names}")

            except Exception as e:
                logger.error(f"Error loading ONNX model: {e}")

                # If this is a quantized model that failed to load, try the regular model
                if "_int8" in model_path:
                    logger.warning("Quantized model failed to load. Falling back to FP32 model.")
                    # Remove the failed INT8 model if it's causing issues
                    try:
                        os.rename(model_path, f"{model_path}.failed")
                        logger.info(f"Renamed problematic quantized model: {model_path}")
                    except Exception as rm_err:
                        logger.warning(f"Could not rename quantized model: {rm_err}")

                    # Try to load the regular FP32 model
                    fp32_path = model_path.replace("_int8", "")
                    if os.path.exists(fp32_path):
                        logger.info(f"Trying to load FP32 model: {fp32_path}")
                        self.ort_session = ort.InferenceSession(
                            fp32_path, providers=execution_providers, sess_options=session_options
                        )
                        self.onnx_path = fp32_path

                        # Cache input/output names
                        self.input_name = self.ort_session.get_inputs()[0].name
                        self.output_names = [o.name for o in self.ort_session.get_outputs()]

                        logger.info(
                            f"FP32 model loaded successfully with providers: {self.ort_session.get_providers()}"
                        )
                    else:
                        raise ValueError(f"FP32 model not found at {fp32_path}")
                else:
                    raise

        except ImportError:
            logger.error(
                "ONNX Runtime not installed. Please install with 'pip install onnxruntime'"
            )
            raise
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            raise

    def initialize_with_warmup(self) -> None:
        """
        Perform model warm-up for faster first inference by running
        through the model once with dummy input.
        """
        if self.ort_session is None:
            logger.warning("No ONNX session loaded, skipping warmup")
            return

        try:
            logger.info("Warming up ONNX model...")
            # Generate dummy input similar to real audio features
            dummy_input = np.zeros((1, 80, 3000), dtype=np.float32)

            # Run inference on dummy input to warm up the model
            ort_inputs = {self.input_name: dummy_input}
            _ = self.ort_session.run(self.output_names, ort_inputs)
            logger.info("ONNX model warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
            # Continue even if warmup fails

    def _preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int, float]:
        """
        Optimized audio preprocessing using faster libraries when available

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate, duration)
        """
        try:
            # Use soundfile for faster loading if available
            if HAS_SOUNDFILE:
                waveform, sample_rate = sf.read(audio_path, dtype="float32")

                # Handle stereo to mono conversion
                if len(waveform.shape) > 1 and waveform.shape[1] > 1:
                    waveform = np.mean(waveform, axis=1)

                # Ensure numpy array is contiguous in memory for faster processing
                waveform = np.ascontiguousarray(waveform)

                # Resample if needed with faster algorithm
                if sample_rate != 16000:
                    if HAS_RESAMPY:
                        waveform = resampy.resample(
                            waveform, sample_rate, 16000, filter="kaiser_fast"
                        )
                    else:
                        # Use torchaudio as fallback
                        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
                        transform = torchaudio.transforms.Resample(sample_rate, 16000)
                        waveform_tensor = transform(waveform_tensor)
                        waveform = waveform_tensor.squeeze(0).numpy()

                    sample_rate = 16000

                # Calculate duration
                duration = len(waveform) / sample_rate

                return waveform, sample_rate, duration
            else:
                # Fallback to torchaudio
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

                # Convert to numpy for consistent interface
                waveform = waveform.squeeze().numpy()

                return waveform, sample_rate, duration

        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            # Return None values to indicate failure
            return None, None, None

    def _optimized_ctc_decode(self, logits: np.ndarray) -> str:
        """
        Optimized CTC decoding with numba JIT compilation if available

        Args:
            logits: Model output logits

        Returns:
            Decoded text
        """
        # Get the most likely token at each timestamp (argmax along vocab dimension)
        predicted_ids = np.argmax(logits[0], axis=-1)
        pad_token_id = self.processor.tokenizer.pad_token_id

        # If numba is available, use JIT-compiled function for faster processing
        if HAS_NUMBA:

            @jit(nopython=True)
            def _collapse_repeated(ids, blank_id):
                """JIT-compiled function for collapsing repeated tokens"""
                result = []
                prev_id = -1
                for id in ids:
                    if id != blank_id and id != prev_id:
                        result.append(id)
                    prev_id = id
                return result

            # Filter out pad tokens and collapse repeats in one pass
            ids_no_blank = predicted_ids[predicted_ids != pad_token_id]
            collapsed_ids = _collapse_repeated(ids_no_blank, pad_token_id)

        else:
            # Fallback to numpy operations which are still efficient
            # Get mask for non-blank tokens
            non_blank_mask = predicted_ids != pad_token_id
            filtered_ids = predicted_ids[non_blank_mask]

            # Find where consecutive tokens differ (for collapsing repeats)
            if len(filtered_ids) > 0:
                # Append a sentinel value to detect the last change
                padded_ids = np.append(filtered_ids, -1)
                # Find where consecutive values are different (including the sentinel)
                changes = np.where(padded_ids[1:] != padded_ids[:-1])[0]
                # Get the values at change indices
                collapsed_ids = filtered_ids[changes]
            else:
                collapsed_ids = filtered_ids

        # Convert to list for tokenizer
        if type(collapsed_ids) is not list:
            token_ids = collapsed_ids.tolist() if HAS_NUMBA else collapsed_ids.tolist()
        else:
            token_ids = collapsed_ids

        # Decode to text
        transcription = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)

        # Remove the first two characters (tokenization bug fix)
        if len(transcription) > 2:
            transcription = transcription[2:]

        return transcription

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file with optimized processing pipeline

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with transcription result and metadata
        """
        start_time = time.time()

        try:
            # Check if ONNX is available
            if self.ort_session is None:
                raise ValueError("ONNX session not initialized properly")

            # Use optimized preprocessing
            waveform, sample_rate, duration = self._preprocess_audio(audio_path)

            if waveform is None:
                raise ValueError("Audio preprocessing failed")

            # Extract features with processor
            inputs = self.processor.feature_extractor(
                waveform, sampling_rate=sample_rate, return_tensors="np"
            )

            # Run inference with ONNX Runtime
            ort_inputs = {self.input_name: inputs.input_features}

            # Get logits from ONNX model
            logits = self.ort_session.run(self.output_names, ort_inputs)[0]

            # Use optimized CTC decoding
            transcription = self._optimized_ctc_decode(logits)

            # Calculate processing time and real-time factor
            processing_time = time.time() - start_time
            real_time_factor = processing_time / duration if duration > 0 else 0

            # Return results with metadata
            return {
                "text": transcription,
                "duration": duration,
                "processing_time": processing_time,
                "real_time_factor": real_time_factor,
            }

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            import traceback

            logger.error(traceback.format_exc())

            # Return error information
            return {
                "text": f"Error: {str(e)}",
                "duration": 0.0,
                "processing_time": time.time() - start_time,
                "error": True,
            }

    def transcribe_batch(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Batch transcription for multiple audio files

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of transcription results
        """
        if not audio_paths:
            return []

        batch_start_time = time.time()
        results = []

        try:
            # Preprocess all audio files
            preprocessed = [self._preprocess_audio(path) for path in audio_paths]

            # Create batch input
            batch_features = []
            valid_indices = []

            for i, (waveform, sample_rate, _) in enumerate(preprocessed):
                if waveform is not None:
                    features = self.processor.feature_extractor(
                        waveform, sampling_rate=sample_rate, return_tensors="np"
                    ).input_features
                    batch_features.append(features)
                    valid_indices.append(i)

            if not batch_features:
                logger.warning("No valid audio files in batch")
                # Return error for each file
                return [
                    {"text": "Error: Audio preprocessing failed", "duration": 0.0, "error": True}
                    for _ in audio_paths
                ]

            # Stack features for batch processing
            batched_input = np.vstack(batch_features)

            # Run inference once for the whole batch
            ort_inputs = {self.input_name: batched_input}
            batched_logits = self.ort_session.run(self.output_names, ort_inputs)[0]

            # Initialize results list with None placeholders
            batch_results = [None] * len(audio_paths)

            # Process each result separately
            for batch_idx, original_idx in enumerate(valid_indices):
                waveform, sample_rate, duration = preprocessed[original_idx]

                # Get logits for this sample
                logits = batched_logits[batch_idx : batch_idx + 1]

                # Decode transcription
                transcription = self._optimized_ctc_decode(logits)

                # Calculate processing metrics
                end_time = time.time()
                processing_time = end_time - batch_start_time
                real_time_factor = processing_time / duration if duration > 0 else 0

                # Store result
                batch_results[original_idx] = {
                    "text": transcription,
                    "duration": duration,
                    "processing_time": processing_time,
                    "real_time_factor": real_time_factor,
                }

            # Fill in any missing results (for failed preprocessing)
            for i in range(len(audio_paths)):
                if batch_results[i] is None:
                    batch_results[i] = {
                        "text": "Error: Audio preprocessing failed",
                        "duration": 0.0,
                        "processing_time": 0.0,
                        "error": True,
                    }

            return batch_results

        except Exception as e:
            logger.error(f"Error in batch transcription: {e}")
            import traceback

            logger.error(traceback.format_exc())

            # Fall back to individual processing
            return [self.transcribe(path) for path in audio_paths]

    def _convert_to_onnx(self) -> Dict[str, str]:
        """
        Convert PyTorch model to ONNX with advanced optimization

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
                logger.info(f"INT8 ONNX model already exists at {int8_onnx_path}")
                return {"model": int8_onnx_path}

            # Check if FP32 model already exists
            fp32_onnx_path = os.path.join(onnx_dir, "model.onnx")
            if os.path.exists(fp32_onnx_path):
                logger.info(f"FP32 ONNX model exists at {fp32_onnx_path}")

                # Try to optimize the existing model with onnxruntime
                try:
                    from onnxruntime.transformers import optimizer

                    optimized_path = os.path.join(onnx_dir, "model_optimized.onnx")

                    # Apply graph optimizations
                    opt_model = optimizer.optimize_model(
                        fp32_onnx_path,
                        model_type="whisper",
                        num_heads=8,  # Adjust based on tiny model
                        hidden_size=384,  # Adjust based on tiny model
                    )
                    opt_model.save_model_to_file(optimized_path)
                    logger.info(f"ONNX model optimized: {optimized_path}")

                    # Try INT8 quantization
                    try:
                        from onnxruntime.quantization import quantize_dynamic, QuantType

                        logger.info(f"Applying INT8 quantization to ONNX model...")
                        quantize_dynamic(
                            model_input=optimized_path,
                            model_output=int8_onnx_path,
                            per_channel=True,
                            weight_type=QuantType.QUInt8,
                            optimization_level=99,
                            op_types_to_quantize=["MatMul", "Gemm", "Conv", "Relu"],
                        )
                        logger.info(f"INT8 quantization completed: {int8_onnx_path}")

                        if os.path.exists(int8_onnx_path):
                            return {"model": int8_onnx_path}
                    except Exception as quant_error:
                        logger.warning(f"INT8 quantization failed: {quant_error}")

                    return {"model": optimized_path}

                except ImportError:
                    logger.warning("ONNX Runtime optimizer not available")
                except Exception as opt_error:
                    logger.warning(f"ONNX optimization failed: {opt_error}")

                # Return original model if optimization failed
                return {"model": fp32_onnx_path}

            # If no existing models, we need to convert from PyTorch
            # This part requires access to the original model implementation
            # For this example, I'll assume there's an ASRInferenceModel class we can use

            logger.info("Converting PyTorch model to ONNX format...")

            # Import the original model
            from src.models.inference_model import ASRInferenceModel

            # Load PyTorch model
            pytorch_model = ASRInferenceModel(self.model_id, self.device)

            # Ensure the model is loaded
            if pytorch_model.model is None:
                raise ValueError("Failed to load PyTorch model for conversion")

            # Create dummy input for tracing
            dummy_input = torch.randn(1, 80, 3000).to(self.device)

            # Set model to eval mode
            pytorch_model.model.eval()

            logger.info(f"Exporting model to ONNX at {fp32_onnx_path}")

            # Export the model to ONNX with enhanced settings
            with torch.no_grad():
                torch.onnx.export(
                    pytorch_model.model,
                    dummy_input,
                    fp32_onnx_path,
                    input_names=["input_features"],
                    output_names=["logits"],
                    dynamic_axes={
                        "input_features": {0: "batch_size", 2: "sequence_length"},
                        "logits": {0: "batch_size", 1: "sequence_length"},
                    },
                    opset_version=14,
                    do_constant_folding=True,  # Fold constants for optimization
                    verbose=False,
                )

            logger.info(f"Model exported to ONNX successfully: {fp32_onnx_path}")

            # Try to optimize and quantize the model
            try:
                # Verify the exported model
                import onnx

                onnx_model = onnx.load(fp32_onnx_path)
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX model verification passed")

                # Apply optimization
                try:
                    from onnxruntime.transformers import optimizer

                    optimized_path = os.path.join(onnx_dir, "model_optimized.onnx")

                    # Apply graph optimizations
                    opt_model = optimizer.optimize_model(
                        fp32_onnx_path, model_type="whisper", num_heads=8, hidden_size=384
                    )
                    opt_model.save_model_to_file(optimized_path)
                    logger.info(f"ONNX model optimized: {optimized_path}")
                    fp32_onnx_path = optimized_path
                except ImportError:
                    logger.warning("ONNX Runtime optimizer not available")
                except Exception as opt_error:
                    logger.warning(f"ONNX optimization failed: {opt_error}")

                # Apply INT8 quantization
                try:
                    from onnxruntime.quantization import quantize_dynamic, QuantType

                    logger.info(f"Applying INT8 quantization to ONNX model...")
                    quantize_dynamic(
                        model_input=fp32_onnx_path,
                        model_output=int8_onnx_path,
                        per_channel=True,
                        weight_type=QuantType.QUInt8,
                        optimization_level=99,
                        op_types_to_quantize=["MatMul", "Gemm", "Conv", "Relu"],
                    )
                    logger.info(f"INT8 quantization completed: {int8_onnx_path}")

                    if os.path.exists(int8_onnx_path):
                        return {"model": int8_onnx_path}
                except ImportError:
                    logger.warning("ONNX Runtime quantization tools not available")
                except Exception as quant_error:
                    logger.warning(f"INT8 quantization failed: {quant_error}")

            except ImportError:
                logger.warning("ONNX package not installed")
            except Exception as verify_error:
                logger.warning(f"ONNX model verification failed: {verify_error}")

            # Return the model files (even if optimization failed)
            return {"model": fp32_onnx_path}

        except Exception as e:
            logger.error(f"Error converting model to ONNX: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {}


# Usage example
if __name__ == "__main__":
    # Configure logging
    import sys

    # Create handler with a more detailed formatter
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Get the root logger and set its handler
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)

    # Create and initialize the model
    model = OptimizedONNXASRInferenceModel(
        model_id="tuandunghcmut/PhoWhisper-tiny-CTC", device="cpu", num_threads=4
    )

    # Example: transcribe a single file
    result = model.transcribe("examples/vietnamese_example_1.mp3")
    print(f"Transcription: {result['text']}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Processing time: {result['processing_time']:.2f}s")
    print(f"Real-time factor: {result['real_time_factor']:.2f}x")

    # Example: batch transcription
    batch_results = model.transcribe_batch(
        [
            "examples/vietnamese_example_1.mp3",
            "examples/vietnamese_example_2.mp3",
        ]
    )

    for i, res in enumerate(batch_results):
        print(f"\nBatch item {i+1}:")
        print(f"Transcription: {res['text']}")
        print(f"Real-time factor: {res.get('real_time_factor', 0):.2f}x")
