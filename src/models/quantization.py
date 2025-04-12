from typing import Optional
import torch
import bitsandbytes as bnb
from ..core.system_logging import logger

class QuantizationManager:
    """Manages model quantization settings and operations."""
    
    def __init__(self):
        self.current_mode: Optional[str] = None
    
    def set_quantization_mode(self, mode: str) -> None:
        """
        Set the quantization mode for models.
        
        Args:
            mode: Quantization mode ('fp16', 'int8', 'int4', or None)
        """
        try:
            valid_modes = ['fp16', 'int8', 'int4', None]
            if mode not in valid_modes:
                raise ValueError(f"Invalid quantization mode. Must be one of {valid_modes}")
            
            self.current_mode = mode
            logger.log_info(f"Quantization mode set to: {mode}")
            
        except Exception as e:
            logger.log_error(f"Error setting quantization mode: {e}")
            raise
    
    def get_quantization_dtype(self) -> torch.dtype:
        """
        Get the torch dtype corresponding to the current quantization mode.
        
        Returns:
            torch.dtype for the current quantization mode
        """
        if self.current_mode == 'fp16':
            return torch.float16
        elif self.current_mode == 'int8':
            return torch.int8
        elif self.current_mode == 'int4':
            return torch.int8  # int4 is handled differently
        else:
            return torch.float32
    
    def quantize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply quantization to a model based on the current mode.
        
        Args:
            model: The model to quantize
            
        Returns:
            The quantized model
        """
        try:
            if self.current_mode == 'fp16':
                return model.half()
            elif self.current_mode == 'int8':
                return bnb.nn.Quant8BitLinear.from_float(model)
            elif self.current_mode == 'int4':
                return bnb.nn.Quant4BitLinear.from_float(model)
            else:
                return model
        except Exception as e:
            logger.log_error(f"Error quantizing model: {e}")
            raise

# Create a singleton instance
quantization_manager = QuantizationManager() 