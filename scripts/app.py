#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse

# Add source directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for app."""
    parser = argparse.ArgumentParser(description="Run Vietnamese ASR Demo App")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to model checkpoint (if not using HuggingFace)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        choices=["cpu", "cuda"], 
        help="Device to run inference on"
    )
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Share the app publicly through Gradio"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Port to run the app on"
    )
    parser.add_argument(
        "--app", 
        type=str, 
        default="gradio",
        choices=["gradio", "streamlit"], 
        help="Which app to run (gradio or streamlit)"
    )
    
    return parser.parse_args()

def main():
    """Run the demo app."""
    # Parse arguments
    args = parse_args()
    
    # Set environment variables from arguments
    os.environ["INFERENCE_DEVICE"] = args.device
    os.environ["PORT"] = str(args.port)
    
    if args.checkpoint:
        os.environ["MODEL_CHECKPOINT_PATH"] = args.checkpoint
        
    if args.share:
        os.environ["GRADIO_SHARE"] = "true"
    
    # Run the appropriate app
    if args.app == "gradio":
        logger.info("Starting Gradio demo app...")
        from src.app.gradio_demo import main as gradio_main
        gradio_main()
    elif args.app == "streamlit":
        logger.info("Starting Streamlit app...")
        logger.info("To run the Streamlit app, use: cd ui && streamlit run app.py")
        import subprocess
        subprocess.run(["streamlit", "run", "ui/app.py"], check=True)
    else:
        logger.error(f"Unknown app type: {args.app}")
        sys.exit(1)

if __name__ == "__main__":
    main() 