#!/usr/bin/env python
"""
Vietnamese ASR Application Launcher

This script serves as a unified entry point for the Vietnamese ASR application,
allowing users to run different components of the system, including:
- Inference with the ASR model
- Interactive demo applications (Gradio or Streamlit)
- API server
"""

import os
import sys
import time
import argparse
import logging
import subprocess
from typing import Optional, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run")

def run_app(args: argparse.Namespace) -> int:
    """Run the interactive demo app"""
    if args.app.lower() == "gradio":
        logger.info(f"Starting Gradio demo on port {args.port}...")
        from app import load_model, create_gradio_app
        
        # Load the model
        model = load_model(args.device)
        
        # Create and run the Gradio app
        app = create_gradio_app(model)
        app.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            debug=args.debug
        )
        return 0
    
    elif args.app.lower() == "streamlit":
        logger.info(f"Starting Streamlit UI...")
        streamlit_path = "ui/app.py"
        if not os.path.exists(streamlit_path):
            logger.error(f"Streamlit app not found at {streamlit_path}")
            return 1
        
        cmd = [
            "streamlit", "run", streamlit_path, 
            "--server.port", str(args.port),
            "--server.address", "0.0.0.0"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        return subprocess.call(cmd)
    
    elif args.app.lower() == "api":
        logger.info(f"Starting API server on port {args.port}...")
        os.environ["PORT"] = str(args.port)
        os.environ["INFERENCE_DEVICE"] = args.device
        
        from api.app import app
        import uvicorn
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=args.port,
            log_level="info"
        )
        return 0
    
    else:
        logger.error(f"Unknown app type: {args.app}")
        return 1

def infer(args: argparse.Namespace) -> int:
    """Run inference on audio files"""
    logger.info(f"Running inference on {args.audio}...")
    
    try:
        from src.models.inference_model import create_asr_model
        import torchaudio
        import glob
        import os
        
        # Load model
        model = create_asr_model(args.model, args.device)
        logger.info(f"Model loaded successfully on {args.device}")
        
        # Prepare output file
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        output_file = open(args.output, "w", encoding="utf-8")
        
        # Process input (file or directory)
        files_to_process = []
        if os.path.isdir(args.audio):
            # Process all audio files in directory
            for ext in [".wav", ".mp3", ".flac", ".ogg"]:
                files_to_process.extend(glob.glob(os.path.join(args.audio, f"**/*{ext}"), recursive=True))
        else:
            # Process single file
            files_to_process = [args.audio]
        
        logger.info(f"Found {len(files_to_process)} file(s) to process")
        
        # Process each file
        for i, audio_file in enumerate(files_to_process):
            try:
                logger.info(f"Processing {i+1}/{len(files_to_process)}: {audio_file}")
                
                # Measure processing time
                start_time = time.time()
                
                # Transcribe
                result = model.transcribe(audio_file)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Write to output
                output_file.write(f"File: {audio_file}\n")
                output_file.write(f"Text: {result['text']}\n")
                output_file.write(f"Duration: {result.get('duration', 'N/A')}s\n")
                output_file.write(f"Processing time: {processing_time:.2f}s\n")
                if 'duration' in result:
                    rtf = processing_time / result['duration']
                    output_file.write(f"Real-time factor: {rtf:.2f}x\n")
                output_file.write("\n---\n\n")
                
                # Also print to console
                logger.info(f"Transcription: {result['text']}")
                logger.info(f"Processing time: {processing_time:.2f}s")
                if 'duration' in result:
                    logger.info(f"Real-time factor: {rtf:.2f}x")
                
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                output_file.write(f"File: {audio_file}\n")
                output_file.write(f"Error: {str(e)}\n\n---\n\n")
        
        output_file.close()
        logger.info(f"Results written to {args.output}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return 1

def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vietnamese ASR Application")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # App parser
    app_parser = subparsers.add_parser("app", help="Run interactive demo")
    app_parser.add_argument("--app", type=str, default="gradio", 
                           choices=["gradio", "streamlit", "api"],
                           help="App type to run (gradio, streamlit, or api)")
    app_parser.add_argument("--port", type=int, default=7860, 
                           help="Port to run the app on")
    app_parser.add_argument("--device", type=str, default="cpu",
                           help="Device to run inference on (cpu or cuda)")
    app_parser.add_argument("--share", action="store_true",
                           help="Share the app publicly")
    app_parser.add_argument("--debug", action="store_true",
                           help="Run in debug mode")
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference on audio files")
    infer_parser.add_argument("--audio", type=str, required=True,
                             help="Path to audio file or directory")
    infer_parser.add_argument("--output", type=str, default="outputs/transcriptions.txt",
                             help="Path to save transcription results")
    infer_parser.add_argument("--model", type=str, 
                             default="tuandunghcmut/PhoWhisper-tiny-CTC",
                             help="Model to use for inference")
    infer_parser.add_argument("--device", type=str, default="cpu",
                             help="Device to run inference on (cpu or cuda)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "app":
        return run_app(args)
    elif args.command == "infer":
        return infer(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 