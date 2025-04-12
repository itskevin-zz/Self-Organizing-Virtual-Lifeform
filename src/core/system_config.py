import json
from typing import Any, Dict, Optional
from pathlib import Path
import os

class ConfigurationError(Exception):
    """Raised when there is an error in the configuration."""
    pass

def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Retrieve a configuration value with a default fallback.
    
    Args:
        config: The configuration dictionary
        key: The configuration key to retrieve
        default: Default value if key is not found
        
    Returns:
        The configuration value or default if not found
    """
    return config.get(key, default)

def load_config(file_path: str = "config.JSON") -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        ConfigurationError: If the file cannot be loaded or parsed
    """
    try:
        if not os.path.exists(file_path):
            raise ConfigurationError(f"Configuration file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            config = json.load(f)
            
        _validate_config(config)
        return config
        
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration: {e}")

def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration structure and values.
    
    Args:
        config: The configuration dictionary to validate
        
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    required_keys = [
        "model_name",
        "scaffold_model_name",
        "max_seq_length",
        "batch_size"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ConfigurationError(f"Missing required configuration key: {key}")
            
    if not isinstance(config.get("max_seq_length"), int) or config["max_seq_length"] <= 0:
        raise ConfigurationError("max_seq_length must be a positive integer")
        
    if not isinstance(config.get("batch_size"), int) or config["batch_size"] <= 0:
        raise ConfigurationError("batch_size must be a positive integer")

class ConfigManager:
    """Manages system configuration loading and validation."""
    
    def __init__(self, config_path: str = "config.JSON"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file: {self.config_path}")
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        try:
            value = self.config
            for k in key.split("."):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def validate_config(self) -> None:
        """Validate the configuration structure and values."""
        required_sections = ["core_config", "lora_config", "training_config", "controls_config"]
        
        # Check required sections
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate core configuration
        core_config = self.config["core_config"]
        if not isinstance(core_config.get("base_model_name"), str):
            raise ValueError("base_model_name must be a string")
        if not isinstance(core_config.get("scaffold_model_name"), str):
            raise ValueError("scaffold_model_name must be a string")
        
        # Validate training configuration
        training_config = self.config["training_config"]
        if not isinstance(training_config.get("learning_rate"), (int, float)):
            raise ValueError("learning_rate must be a number")
        if not isinstance(training_config.get("train_epochs"), int):
            raise ValueError("train_epochs must be an integer")
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save the current configuration to a file."""
        save_path = path or self.config_path
        with open(save_path, "w") as f:
            json.dump(self.config, f, indent=2)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration values."""
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
        self.validate_config()

# Create a singleton instance
config_manager = ConfigManager() 