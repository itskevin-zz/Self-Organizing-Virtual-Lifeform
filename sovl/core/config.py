import json
import os
from typing import Any, Dict, Optional

# Configuration constants
DEFAULT_CONFIG = {
    "model": {
        "base_model": "mistralai/Mistral-7B-v0.1",
        "scaffold_model": "mistralai/Mistral-7B-v0.1",
        "max_seq_length": 2048,
        "batch_size": 4,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "num_train_epochs": 3
    },
    "memory": {
        "token_map_size": 1000,
        "dream_memory_size": 100,
        "memory_decay_rate": 0.95,
        "prune_threshold": 0.1
    },
    "curiosity": {
        "spontaneous_threshold": 0.7,
        "response_threshold": 0.5,
        "pressure_threshold": 0.8,
        "pressure_drop": 0.1,
        "silence_threshold": 300,
        "question_cooldown": 60,
        "queue_maxlen": 10
    },
    "temperament": {
        "eager_threshold": 0.7,
        "sluggish_threshold": 0.3,
        "mood_influence": 0.5,
        "curiosity_boost": 0.2,
        "restless_drop": 0.1,
        "melancholy_noise": 0.1
    }
}

def get_config_value(config: Dict[str, Any], key: str, default: Optional[Any] = None) -> Any:
    """
    Get a configuration value using dot notation.
    
    Args:
        config: The configuration dictionary
        key: The key to retrieve (can use dot notation for nested keys)
        default: Default value if key is not found
        
    Returns:
        The configuration value or default if not found
    """
    try:
        keys = key.split('.')
        value = config
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default

def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and ensure all required configuration values are present.
    
    Args:
        config: The configuration dictionary to validate
        
    Returns:
        Validated configuration dictionary
    """
    validated_config = DEFAULT_CONFIG.copy()
    
    # Update with provided values
    def update_dict(base: Dict[str, Any], update: Dict[str, Any]) -> None:
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                update_dict(base[key], value)
            else:
                base[key] = value
    
    update_dict(validated_config, config)
    return validated_config

def load_config(config_path: str = "config.JSON") -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Loaded and validated configuration dictionary
    """
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return _validate_config(config)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading config: {e}")
        return DEFAULT_CONFIG 