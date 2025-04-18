#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse

# Add source directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from source.utils.logging import setup_logging, log_system_info

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Vietnamese ASR with CTC")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml", 
        help="Path to training configuration file"
    )
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--audio", 
        type=str, 
        required=True,
        help="Path to audio file or directory"
    )
    infer_parser.add_argument(
        "--output", 
        type=str, 
        default="outputs/transcriptions.txt", 
        help="Path to output file"
    )
    infer_parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to model checkpoint"
    )
    infer_parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="Device to run inference on"
    )
    
    # App command
    app_parser = subparsers.add_parser("app", help="Run Gradio app")
    app_parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to model checkpoint"
    )
    app_parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="Device to run inference on"
    )
    app_parser.add_argument(
        "--share", 
        action="store_true", 
        help="Share the app publicly"
    )
    app_parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Port to run the app on"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging()
    
    # Log system information
    log_system_info()
    
    # Run the appropriate command
    if args.command == "train":
        # Import here to avoid loading unnecessary modules
        from scripts.train import main as train_main
        # Convert namespace to dict and pass to train
        sys.argv = [sys.argv[0], args.config]
        train_main()
    
    elif args.command == "infer":
        # Import here to avoid loading unnecessary modules
        from scripts.inference import main as infer_main
        # Manually set sys.argv to pass to inference script
        sys.argv = [
            sys.argv[0],
            "--audio", args.audio,
            "--output", args.output,
        ]
        if args.checkpoint:
            sys.argv.extend(["--checkpoint", args.checkpoint])
        if args.device:
            sys.argv.extend(["--device", args.device])
        
        infer_main()
    
    elif args.command == "app":
        # Import here to avoid loading unnecessary modules
        from scripts.app import main as app_main
        # Manually set sys.argv to pass to app script
        sys.argv = [sys.argv[0]]
        if args.checkpoint:
            sys.argv.extend(["--checkpoint", args.checkpoint])
        if args.device:
            sys.argv.extend(["--device", args.device])
        if args.share:
            sys.argv.append("--share")
        sys.argv.extend(["--port", str(args.port)])
        
        app_main()
    
    else:
        print("Please specify a command: train, infer, or app")
        print("For help, run: python run.py -h")

if __name__ == "__main__":
    main() 