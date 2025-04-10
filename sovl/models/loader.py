from typing import Dict, Any, Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb

def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    use_4bit: bool = False,
    use_8bit: bool = False,
    use_lora: bool = False,
    lora_config: Optional[Dict[str, Any]] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and its tokenizer with optional quantization and LoRA.
    
    Args:
        model_name: Name or path of the model to load
        device: Device to load the model on
        use_4bit: Whether to use 4-bit quantization
        use_8bit: Whether to use 8-bit quantization
        use_lora: Whether to use LoRA
        lora_config: Configuration for LoRA if used
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure model loading
    load_kwargs = {}
    if use_4bit:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["quantization_config"] = bnb.QuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif use_8bit:
        load_kwargs["load_in_8bit"] = True
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        **load_kwargs
    )
    
    # Apply LoRA if requested
    if use_lora:
        if lora_config is None:
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM
            }
        
        peft_config = LoraConfig(**lora_config)
        model = get_peft_model(model, peft_config)
    
    return model, tokenizer

def initialize_models(
    base_model_name: str,
    scaffold_model_name: str,
    config: Dict[str, Any]
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, PreTrainedModel, PreTrainedTokenizer]:
    """
    Initialize both base and scaffold models with their tokenizers.
    
    Args:
        base_model_name: Name or path of the base model
        scaffold_model_name: Name or path of the scaffold model
        config: Configuration dictionary
        
    Returns:
        Tuple of (base_model, base_tokenizer, scaffold_model, scaffold_tokenizer)
    """
    # Load base model
    base_model, base_tokenizer = load_model_and_tokenizer(
        base_model_name,
        use_4bit=config.get("use_4bit", False),
        use_8bit=config.get("use_8bit", False),
        use_lora=config.get("use_lora", False),
        lora_config=config.get("lora_config")
    )
    
    # Load scaffold model
    scaffold_model, scaffold_tokenizer = load_model_and_tokenizer(
        scaffold_model_name,
        use_4bit=config.get("scaffold_use_4bit", False),
        use_8bit=config.get("scaffold_use_8bit", False)
    )
    
    return base_model, base_tokenizer, scaffold_model, scaffold_tokenizer

def get_model_layers(model: PreTrainedModel) -> list:
    """
    Get all transformer layers from a model.
    
    Args:
        model: The model to get layers from
        
    Returns:
        List of transformer layers
    """
    layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList):
            layers.extend(module)
    return layers 