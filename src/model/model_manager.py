from typing import Optional, Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
from ..core.system_logging import logger

class ModelManager:
    """Manages model loading, initialization, and configuration."""
    
    def __init__(self, base_model_name: str, scaffold_model_name: str, quantization_mode: str = "fp16"):
        self.base_model_name = base_model_name
        self.scaffold_model_name = scaffold_model_name
        self.quantization_mode = quantization_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models and tokenizers
        self.base_model = None
        self.scaffolds = []
        self.base_tokenizer = None
        self.scaffold_tokenizer = None
        self.scaffold_proj = None
        
        # Load configurations
        self.base_config = AutoConfig.from_pretrained(base_model_name)
        self.scaffold_config = AutoConfig.from_pretrained(scaffold_model_name)
        
    def load_models(self) -> None:
        """Load and initialize base and scaffold models."""
        try:
            # Load base model
            quantization_config = self._get_quantization_config()
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                config=self.base_config,
                **quantization_config
            ).to(self.device)
            self.base_model.eval()
            for param in self.base_model.parameters():
                param.requires_grad = False
                
            # Load scaffold model
            scaffold_model_raw = AutoModelForCausalLM.from_pretrained(
                self.scaffold_model_name,
                config=self.scaffold_config,
                **quantization_config
            )
            
            # Apply LoRA if enabled
            if self._should_apply_lora():
                lora_config = self._get_lora_config()
                self.scaffolds = [get_peft_model(scaffold_model_raw, lora_config).to(self.device)]
            else:
                self.scaffolds = [scaffold_model_raw.to(self.device)]
                
            # Load tokenizers
            self._load_tokenizers()
            
            # Initialize projection if needed
            self._initialize_projection()
            
        except Exception as e:
            logger.log_error(f"Error loading models: {e}")
            raise
            
    def _get_quantization_config(self) -> Dict[str, Any]:
        """Get quantization configuration based on mode."""
        if self.quantization_mode == "int8":
            return {"load_in_8bit": True}
        elif self.quantization_mode == "int4":
            return {"load_in_4bit": True}
        return {}
        
    def _should_apply_lora(self) -> bool:
        """Check if LoRA should be applied to scaffold model."""
        # This could be made configurable
        return True
        
    def _get_lora_config(self) -> LoraConfig:
        """Get LoRA configuration."""
        return LoraConfig(
            r=8,  # LoRA rank
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
    def _load_tokenizers(self) -> None:
        """Load and configure tokenizers."""
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.scaffold_tokenizer = AutoTokenizer.from_pretrained(self.scaffold_model_name)
        
        # Set pad tokens if not set
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        if self.scaffold_tokenizer.pad_token is None:
            self.scaffold_tokenizer.pad_token = self.scaffold_tokenizer.eos_token
            
        # Update model configs with pad token IDs
        self.base_model.config.pad_token_id = self.base_tokenizer.pad_token_id
        self.scaffolds[0].config.pad_token_id = self.scaffold_tokenizer.pad_token_id
        
    def _initialize_projection(self) -> None:
        """Initialize projection layer if hidden sizes differ."""
        if self.scaffold_config.hidden_size != self.base_config.hidden_size:
            self.scaffold_proj = torch.nn.Linear(
                self.scaffold_config.hidden_size,
                self.base_config.hidden_size
            ).to(self.device)
            self.scaffold_proj.weight.requires_grad_(True)
            self.scaffold_proj.bias.requires_grad_(True)
            
    def get_model_layers(self, model) -> List[torch.nn.Module]:
        """Get the layers of a model."""
        actual_model = model.base_model if hasattr(model, 'base_model') else model
        if hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'h'):
            return actual_model.transformer.h
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            return actual_model.model.layers
        elif hasattr(actual_model, 'layers'):
            return actual_model.layers
        elif hasattr(actual_model, 'decoder') and hasattr(actual_model.decoder, 'layers'):
            return actual_model.decoder.layers
        raise ValueError(f"Cannot determine layer structure for {actual_model.__class__.__name__}")
        
    def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            if self.scaffold_proj is not None:
                del self.scaffold_proj
            for scaffold in self.scaffolds:
                del scaffold
            del self.base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.log_error(f"Error during cleanup: {e}") 