from dataclasses import dataclass
from typing import Optional, Dict, Any
from transformers import PretrainedConfig

@dataclass
class ModelConfig:
    """Configuration for model management."""
    base_model_name: str
    scaffold_model_name: str
    quantization_mode: str = "none"  # none, 4bit, 8bit
    device: str = "cuda"
    lora_config: Optional[Dict[str, Any]] = None
    projection_config: Optional[Dict[str, Any]] = None
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        valid_quantization_modes = ["none", "4bit", "8bit"]
        if self.quantization_mode not in valid_quantization_modes:
            raise ValueError(
                f"Invalid quantization mode: {self.quantization_mode}. "
                f"Must be one of {valid_quantization_modes}"
            )
        return True
        
    def get_quantization_config(self) -> Dict[str, Any]:
        """Get quantization configuration based on mode."""
        if self.quantization_mode == "none":
            return {}
        elif self.quantization_mode == "4bit":
            return {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "float16"
            }
        elif self.quantization_mode == "8bit":
            return {
                "load_in_8bit": True
            }
        else:
            raise ValueError(f"Unsupported quantization mode: {self.quantization_mode}")
            
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration."""
        if self.lora_config is None:
            return {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            }
        return self.lora_config
        
    def get_projection_config(self) -> Dict[str, Any]:
        """Get projection layer configuration."""
        if self.projection_config is None:
            return {
                "in_features": 4096,  # Default for many models
                "out_features": 4096,
                "bias": True
            }
        return self.projection_config 