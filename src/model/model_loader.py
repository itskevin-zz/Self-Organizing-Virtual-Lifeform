from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from ..core.system_logging import logger

class ModelLoader:
    """Handles model loading and validation."""
    
    @staticmethod
    def load_model(model_name: str, device: str = "cuda", **kwargs) -> AutoModelForCausalLM:
        """Load a model with error handling and validation."""
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            model = model.to(device)
            model.eval()
            return model
        except Exception as e:
            logger.log_error(f"Error loading model {model_name}: {e}")
            raise
            
    @staticmethod
    def load_tokenizer(model_name: str, **kwargs) -> AutoTokenizer:
        """Load a tokenizer with error handling."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            logger.log_error(f"Error loading tokenizer for {model_name}: {e}")
            raise
            
    @staticmethod
    def validate_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> bool:
        """Validate model and tokenizer compatibility."""
        try:
            # Check if model and tokenizer are compatible
            if model.config.vocab_size != len(tokenizer):
                logger.log_warning(
                    f"Model vocab size ({model.config.vocab_size}) "
                    f"does not match tokenizer length ({len(tokenizer)})"
                )
                
            # Test forward pass
            test_input = tokenizer("Test input", return_tensors="pt")
            with torch.no_grad():
                model(**test_input)
                
            return True
        except Exception as e:
            logger.log_error(f"Model validation failed: {e}")
            return False
            
    @staticmethod
    def get_model_config(model_name: str) -> AutoConfig:
        """Get model configuration."""
        try:
            return AutoConfig.from_pretrained(model_name)
        except Exception as e:
            logger.log_error(f"Error getting config for {model_name}: {e}")
            raise 