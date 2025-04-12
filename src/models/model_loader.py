from typing import Tuple, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import LoraConfig, get_peft_model, TaskType
from ..core.system_config import config_manager
from ..core.system_logging import logger

class ModelLoader:
    """Handles loading and configuration of base and scaffold models."""
    
    def __init__(self):
        self.base_model: Optional[PreTrainedModel] = None
        self.scaffold_model: Optional[PreTrainedModel] = None
        self.base_tokenizer: Optional[PreTrainedTokenizer] = None
        self.scaffold_tokenizer: Optional[PreTrainedTokenizer] = None
    
    def load_models(self) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizer]:
        """
        Load and configure both base and scaffold models.
        
        Returns:
            Tuple of (base_model, scaffold_model, base_tokenizer, scaffold_tokenizer)
        """
        try:
            # Load configuration
            core_config = config_manager.get_value("core_config")
            lora_config = config_manager.get_value("lora_config")
            
            # Load base model and tokenizer
            self.base_model, self.base_tokenizer = self._load_single_model(
                core_config["base_model_name"],
                is_scaffold=False
            )
            
            # Load scaffold model and tokenizer with LoRA
            self.scaffold_model, self.scaffold_tokenizer = self._load_single_model(
                core_config["scaffold_model_name"],
                is_scaffold=True,
                lora_config=lora_config
            )
            
            logger.log_info("Models loaded successfully")
            return self.base_model, self.scaffold_model, self.base_tokenizer, self.scaffold_tokenizer
            
        except Exception as e:
            logger.log_error(f"Error loading models: {e}")
            raise
    
    def _load_single_model(
        self,
        model_name: str,
        is_scaffold: bool = False,
        lora_config: Optional[dict] = None
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a single model and its tokenizer.
        
        Args:
            model_name: Name of the model to load
            is_scaffold: Whether this is the scaffold model
            lora_config: LoRA configuration if applicable
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model configuration
            config = AutoConfig.from_pretrained(model_name)
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float16 if config_manager.get_value("core_config.quantization") == "fp16" else torch.float32
            )
            
            # Apply LoRA if this is the scaffold model
            if is_scaffold and lora_config:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_config["lora_rank"],
                    lora_alpha=lora_config["lora_alpha"],
                    lora_dropout=lora_config["lora_dropout"],
                    target_modules=lora_config["lora_target_modules"]
                )
                model = get_peft_model(model, peft_config)
            
            return model, tokenizer
            
        except Exception as e:
            logger.log_error(f"Error loading model {model_name}: {e}")
            raise
    
    def get_model_layers(self, model: PreTrainedModel) -> list:
        """
        Get the transformer layers of a model.
        
        Args:
            model: The model to get layers from
            
        Returns:
            List of transformer layers
        """
        try:
            if hasattr(model, "transformer"):
                return model.transformer.h
            elif hasattr(model, "model"):
                return model.model.layers
            else:
                raise ValueError("Model structure not recognized")
        except Exception as e:
            logger.log_error(f"Error getting model layers: {e}")
            raise

# Create a singleton instance
model_loader = ModelLoader() 