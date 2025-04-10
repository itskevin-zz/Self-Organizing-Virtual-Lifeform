from typing import Dict, Any, Optional
import torch
import bitsandbytes as bnb
from transformers import PreTrainedModel

class QuantizationManager:
    """
    Manager for handling model quantization.
    """
    def __init__(self):
        self.quantization_config = None
        self.current_mode = None

    def set_quantization_mode(
        self,
        mode: str,
        model: Optional[PreTrainedModel] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set the quantization mode for a model.
        
        Args:
            mode: Quantization mode ('4bit', '8bit', or None)
            model: Model to quantize (optional)
            config: Additional quantization configuration (optional)
        """
        self.current_mode = mode
        
        if mode == "4bit":
            self.quantization_config = bnb.QuantizationConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif mode == "8bit":
            self.quantization_config = bnb.QuantizationConfig(
                load_in_8bit=True
            )
        else:
            self.quantization_config = None
        
        if model is not None:
            self._apply_quantization(model)

    def _apply_quantization(self, model: PreTrainedModel) -> None:
        """
        Apply quantization to a model.
        
        Args:
            model: Model to quantize
        """
        if self.quantization_config is None:
            return
        
        if self.current_mode == "4bit":
            model = bnb.quantize_model_4bit(model, self.quantization_config)
        elif self.current_mode == "8bit":
            model = bnb.quantize_model_8bit(model, self.quantization_config)

    def get_quantization_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the current quantization configuration.
        
        Returns:
            Current quantization configuration or None
        """
        return self.quantization_config

    def is_quantized(self) -> bool:
        """
        Check if quantization is currently enabled.
        
        Returns:
            True if quantization is enabled, False otherwise
        """
        return self.current_mode is not None

    def get_memory_usage(self, model: PreTrainedModel) -> Dict[str, float]:
        """
        Get memory usage statistics for a quantized model.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary containing memory usage statistics
        """
        if not self.is_quantized():
            return {
                "total_params": sum(p.numel() for p in model.parameters()),
                "memory_bytes": sum(p.numel() * p.element_size() for p in model.parameters())
            }
        
        if self.current_mode == "4bit":
            bits_per_param = 4
        else:  # 8bit
            bits_per_param = 8
        
        total_params = sum(p.numel() for p in model.parameters())
        memory_bytes = (total_params * bits_per_param) / 8
        
        return {
            "total_params": total_params,
            "memory_bytes": memory_bytes,
            "bits_per_param": bits_per_param
        } 