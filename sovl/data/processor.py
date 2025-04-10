from typing import List, Dict, Any, Tuple
import torch
from transformers import PreTrainedTokenizer

def tokenize_and_map(
    prompts: List[str],
    base_tokenizer: PreTrainedTokenizer,
    scaffold_tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    padding: str = 'max_length'
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Tokenize prompts using both base and scaffold tokenizers.
    
    Args:
        prompts: List of prompt strings
        base_tokenizer: Tokenizer for the base model
        scaffold_tokenizer: Tokenizer for the scaffold model
        max_length: Maximum sequence length
        padding: Padding strategy
        
    Returns:
        Tuple of (base_inputs, scaffold_inputs) containing tokenized inputs
    """
    base_inputs = base_tokenizer(
        prompts,
        max_length=max_length,
        padding=padding,
        truncation=True,
        return_tensors="pt"
    )
    
    scaffold_inputs = scaffold_tokenizer(
        prompts,
        max_length=max_length,
        padding=padding,
        truncation=True,
        return_tensors="pt"
    )
    
    return base_inputs, scaffold_inputs

def map_sequence(
    base_input_ids: torch.Tensor,
    token_map: Dict[int, int],
    default_token: int = 0
) -> torch.Tensor:
    """
    Map sequence from base model token space to scaffold model token space.
    
    Args:
        base_input_ids: Input IDs from base model
        token_map: Mapping from base token IDs to scaffold token IDs
        default_token: Default token ID to use for unmapped tokens
        
    Returns:
        Mapped sequence in scaffold token space
    """
    # Create a mapping tensor for efficient lookup
    max_base_token = max(token_map.keys()) if token_map else 0
    mapping_tensor = torch.full((max_base_token + 1,), default_token, dtype=torch.long)
    
    # Fill in the known mappings
    for base_token, scaffold_token in token_map.items():
        mapping_tensor[base_token] = scaffold_token
    
    # Apply the mapping
    return mapping_tensor[base_input_ids]

def _update_token_map_memory(
    prompt: str,
    confidence: float,
    token_map: Dict[int, int],
    base_tokenizer: PreTrainedTokenizer,
    scaffold_tokenizer: PreTrainedTokenizer,
    memory_size: int = 1000,
    decay_rate: float = 0.95
) -> None:
    """
    Update the token mapping memory based on new observations.
    
    Args:
        prompt: Input prompt
        confidence: Confidence score for the mapping
        token_map: Current token mapping
        base_tokenizer: Tokenizer for the base model
        scaffold_tokenizer: Tokenizer for the scaffold model
        memory_size: Maximum size of the token map
        decay_rate: Rate at which to decay old mappings
    """
    # Tokenize the prompt with both tokenizers
    base_tokens = base_tokenizer.encode(prompt, add_special_tokens=False)
    scaffold_tokens = scaffold_tokenizer.encode(prompt, add_special_tokens=False)
    
    # Update mappings for tokens that appear in both sequences
    for base_token, scaffold_token in zip(base_tokens, scaffold_tokens):
        if base_token not in token_map:
            if len(token_map) >= memory_size:
                # Remove the least confident mapping if we're at capacity
                min_confidence_token = min(token_map.items(), key=lambda x: x[1])[0]
                del token_map[min_confidence_token]
            token_map[base_token] = scaffold_token
        else:
            # Decay the existing mapping
            token_map[base_token] = int(
                token_map[base_token] * decay_rate + scaffold_token * (1 - decay_rate)
            ) 