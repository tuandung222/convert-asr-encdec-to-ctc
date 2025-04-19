import argparse
import os
import time
import torch
import logging
from src.models.inference_model import create_asr_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("onnx-test")

def test_onnx_model(audio_path, model_id, device):
    """
    Test the ONNX model implementation by comparing it with PyTorch model
    
    Args:
        audio_path: Path to audio file for testing
        model_id: Model ID or path
        device: Device to run inference on
    """
    logger.info(f"Testing ONNX model implementation...")
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return
    
    # First, run inference using the PyTorch model (Lightning checkpoint)
    logger.info(f"Loading PyTorch model...")
    start_time = time.time()
    pytorch_model = create_asr_model(model_id, device, model_type="pytorch")
    pytorch_load_time = time.time() - start_time
    
    logger.info(f"PyTorch model loaded in {pytorch_load_time:.2f} seconds")
    
    # Run inference with PyTorch model
    logger.info(f"Running inference with PyTorch model...")
    start_time = time.time()
    pytorch_result = pytorch_model.transcribe(audio_path)
    pytorch_inference_time = time.time() - start_time
    
    logger.info(f"PyTorch model inference time: {pytorch_inference_time:.2f} seconds")
    logger.info(f"PyTorch transcription: {pytorch_result.get('text', '')}")
    
    # Now, load the ONNX model
    try:
        logger.info(f"Loading ONNX model...")
        start_time = time.time()
        onnx_model = create_asr_model(model_id, device, model_type="onnx")
        onnx_load_time = time.time() - start_time
        
        logger.info(f"ONNX model loaded in {onnx_load_time:.2f} seconds")
        
        # Run inference with ONNX model
        logger.info(f"Running inference with ONNX model...")
        start_time = time.time()
        onnx_result = onnx_model.transcribe(audio_path)
        onnx_inference_time = time.time() - start_time
        
        logger.info(f"ONNX model inference time: {onnx_inference_time:.2f} seconds")
        logger.info(f"ONNX transcription: {onnx_result.get('text', '')}")
        
        # Compare results
        logger.info(f"Results comparison:")
        logger.info(f"PyTorch: {pytorch_result.get('text', '')}")
        logger.info(f"ONNX:    {onnx_result.get('text', '')}")
        
        # Compare times
        pytorch_total_time = pytorch_load_time + pytorch_inference_time
        onnx_total_time = onnx_load_time + onnx_inference_time
        
        logger.info(f"Time comparison:")
        logger.info(f"PyTorch - Load: {pytorch_load_time:.2f}s, Inference: {pytorch_inference_time:.2f}s, Total: {pytorch_total_time:.2f}s")
        logger.info(f"ONNX    - Load: {onnx_load_time:.2f}s, Inference: {onnx_inference_time:.2f}s, Total: {onnx_total_time:.2f}s")
        logger.info(f"Speed improvement: {pytorch_inference_time/onnx_inference_time:.2f}x faster inference")
        
    except Exception as e:
        logger.error(f"Error testing ONNX model: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ONNX model implementation")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file for testing")
    parser.add_argument("--model", type=str, default="tuandunghcmut/PhoWhisper-tiny-CTC", help="Model ID or path")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda)")
    
    args = parser.parse_args()
    
    test_onnx_model(args.audio, args.model, args.device) 