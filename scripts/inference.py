#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Union, Optional
import time

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_manager import ModelManager
from src.utils.logging import setup_logging, log_system_info
from src.utils.config import setup_config_for_inference
from src.utils.audio import load_audio, get_audio_duration

# Set up logger
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with Vietnamese ASR CTC model")
    
    parser.add_argument(
        "--audio", 
        type=str, 
        help="Path to audio file or directory containing audio files"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="outputs/transcriptions.txt", 
        help="Path to output file for transcriptions"
    )
    
    parser.add_argument(
        "--model_config", 
        type=str, 
        default="configs/model_config.yaml", 
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--inference_config", 
        type=str, 
        default="configs/inference_config.yaml", 
        help="Path to inference configuration file"
    )
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to model checkpoint (overrides config)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        help="Device to run inference on (overrides config)"
    )
    
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        help="Logging level"
    )
    
    return parser.parse_args()

def get_audio_files(path: Union[str, Path]) -> List[Path]:
    """
    Get all audio files from a path (file or directory).
    
    Args:
        path: Path to file or directory
        
    Returns:
        List of audio file paths
    """
    path = Path(path)
    
    if path.is_file():
        return [path]
    
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(list(path.glob(f"**/*{ext}")))
    
    return sorted(audio_files)

def transcribe_files(
    model_manager: ModelManager, 
    audio_files: List[Path], 
    output_file: Optional[str] = None
) -> List[dict]:
    """
    Transcribe a list of audio files.
    
    Args:
        model_manager: Model manager
        audio_files: List of audio file paths
        output_file: Path to output file for transcriptions
        
    Returns:
        List of transcription results
    """
    results = []
    
    # Create output directory if needed
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_file_handle = open(output_file, "w", encoding="utf-8")
    else:
        output_file_handle = None
    
    # Process each file
    for audio_file in audio_files:
        start_time = time.time()
        
        try:
            # Get audio duration
            duration = get_audio_duration(audio_file)
            
            # Transcribe audio
            transcription = model_manager.transcribe_file(audio_file)
            
            # Measure processing time
            process_time = time.time() - start_time
            rtf = process_time / duration  # Real-time factor
            
            # Create result object
            result = {
                "file": str(audio_file),
                "transcription": transcription,
                "duration": duration,
                "process_time": process_time,
                "rtf": rtf
            }
            
            # Log result
            logger.info(f"Transcribed {audio_file.name} ({duration:.2f}s) in {process_time:.2f}s (RTF: {rtf:.2f})")
            logger.info(f"Transcription: {transcription}")
            
            # Write to output file
            if output_file_handle:
                output_file_handle.write(f"File: {audio_file}\n")
                output_file_handle.write(f"Transcription: {transcription}\n")
                output_file_handle.write(f"Duration: {duration:.2f}s, RTF: {rtf:.2f}\n")
                output_file_handle.write("-" * 80 + "\n")
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            if output_file_handle:
                output_file_handle.write(f"File: {audio_file}\n")
                output_file_handle.write(f"Error: {e}\n")
                output_file_handle.write("-" * 80 + "\n")
    
    # Close output file
    if output_file_handle:
        output_file_handle.close()
    
    return results

def main():
    """Main function for inference."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level, log_file="logs/inference.log")
    
    # Log system information
    log_system_info()
    
    # Set up configuration
    override_values = {}
    if args.checkpoint:
        override_values["model.checkpoint_path"] = args.checkpoint
    if args.device:
        override_values["inference.device"] = args.device
    
    config = setup_config_for_inference(
        model_config_path=args.model_config,
        inference_config_path=args.inference_config,
        override_values=override_values
    )
    
    # Create model manager
    model_manager = ModelManager(config)
    
    # Get audio files
    if not args.audio:
        logger.error("No audio file or directory specified")
        sys.exit(1)
    
    audio_files = get_audio_files(args.audio)
    
    if not audio_files:
        logger.error(f"No audio files found at {args.audio}")
        sys.exit(1)
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Transcribe files
    results = transcribe_files(model_manager, audio_files, args.output)
    
    # Print summary
    total_duration = sum(r["duration"] for r in results)
    total_process_time = sum(r["process_time"] for r in results)
    avg_rtf = total_process_time / total_duration if total_duration > 0 else 0
    
    logger.info(f"Transcription complete!")
    logger.info(f"Processed {len(results)} files")
    logger.info(f"Total audio duration: {total_duration:.2f}s")
    logger.info(f"Total processing time: {total_process_time:.2f}s")
    logger.info(f"Average RTF: {avg_rtf:.2f}")
    logger.info(f"Transcriptions saved to: {args.output}")

if __name__ == "__main__":
    main() 