import os
import logging
from typing import Dict, Any, Optional
import yaml
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from a YAML file using OmegaConf.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)
    return config

def save_config(config: DictConfig, output_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration object
        output_path: Path to save the configuration
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save config
    with open(output_path, 'w') as f:
        OmegaConf.save(config, f)
    
    logger.info(f"Config saved to {output_path}")

def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged_config = OmegaConf.merge(base_config, override_config)
    return merged_config

def get_config_dict(config: DictConfig) -> Dict[str, Any]:
    """
    Convert OmegaConf config to a plain dictionary.
    
    Args:
        config: OmegaConf configuration
        
    Returns:
        Configuration as a dictionary
    """
    return OmegaConf.to_container(config, resolve=True)

def setup_config_for_inference(
    model_config_path: str = "configs/model_config.yaml",
    inference_config_path: str = "configs/inference_config.yaml",
    override_values: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Set up configuration for inference by merging model and inference configs.
    
    Args:
        model_config_path: Path to the model configuration file
        inference_config_path: Path to the inference configuration file
        override_values: Additional values to override in the configuration
        
    Returns:
        Merged configuration
    """
    # Load model config
    model_config = load_config(model_config_path)
    
    # Load inference config
    inference_config = load_config(inference_config_path)
    
    # Merge configs
    config = merge_configs(model_config, inference_config)
    
    # Apply override values if provided
    if override_values:
        override_config = OmegaConf.create(override_values)
        config = merge_configs(config, override_config)
    
    return config 