#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

# Add parent directory to path to import source modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.training.trainer import TrainingManager
from source.utils.logging import setup_logging, log_system_info
from source.utils.config import save_config

# Set up logger
logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="training_config")
def main(config: DictConfig) -> None:
    """
    Main training script.
    
    Args:
        config: Hydra configuration
    """
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"train_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_level="INFO", log_file=os.path.join(log_dir, "train.log"))
    
    # Log system information
    log_system_info()
    
    # Log configuration
    logger.info("\n" + OmegaConf.to_yaml(config))
    
    # Save configuration
    save_config(config, os.path.join(log_dir, "config.yaml"))
    
    # Create training manager
    trainer = TrainingManager(config)
    
    # Train model
    best_checkpoint_path = trainer.train()
    
    # Test model
    if os.path.exists(best_checkpoint_path):
        logger.info(f"Testing model with checkpoint: {best_checkpoint_path}")
        test_results = trainer.test(best_checkpoint_path)
        logger.info(f"Test results: {test_results}")
    else:
        logger.warning(f"Checkpoint {best_checkpoint_path} not found. Skipping testing.")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 