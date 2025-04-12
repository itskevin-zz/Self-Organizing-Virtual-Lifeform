from typing import List, Dict, Any, Optional
import torch
from transformers import PreTrainedTokenizer
from ..core.system_logging import logger

class DataProcessor:
    """Handles tokenization and sequence processing for the models."""
    
    def __init__(self, base_tokenizer: PreTrainedTokenizer, scaffold_tokenizer: PreTrainedTokenizer):
        self.base_tokenizer = base_tokenizer
        self.scaffold_tokenizer = scaffold_tokenizer
        self.token_map: Dict[int, int] = {}
        self.token_map_memory: Dict[str, float] = {}
    
    def tokenize_and_map(
        self,
        prompts: List[str],
        max_length: int = 512,
        padding: str = 'max_length'
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize prompts and map them between base and scaffold tokenizers.
        
        Args:
            prompts: List of prompt strings to tokenize
            base_tokenizer: Tokenizer for the base model
            scaffold_tokenizer: Tokenizer for the scaffold model (optional)
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length' or 'longest')
            
        Returns:
            Dictionary containing tokenized inputs
        """
        try:
            # Tokenize with base tokenizer
            base_inputs = self.base_tokenizer(
                prompts,
                max_length=max_length,
                padding=padding,
                truncation=True,
                return_tensors="pt"
            )
            
            result = {
                "base_input_ids": base_inputs["input_ids"],
                "base_attention_mask": base_inputs["attention_mask"]
            }
            
            # If scaffold tokenizer is provided, map the sequences
            if self.scaffold_tokenizer is not None:
                scaffold_inputs = self.scaffold_tokenizer(
                    prompts,
                    max_length=max_length,
                    padding=padding,
                    truncation=True,
                    return_tensors="pt"
                )
                result.update({
                    "scaffold_input_ids": scaffold_inputs["input_ids"],
                    "scaffold_attention_mask": scaffold_inputs["attention_mask"]
                })
            
            return result
        except Exception as e:
            logger.log_error(f"Error in tokenize_and_map: {e}")
            raise
    
    def map_sequence(self, base_input_ids: torch.Tensor) -> torch.Tensor:
        """
        Map sequence from base tokenizer to scaffold tokenizer.
        
        Args:
            base_input_ids: Input IDs from base tokenizer
            
        Returns:
            Mapped input IDs for scaffold tokenizer
        """
        try:
            # Create a mapping tensor for efficient lookup
            max_token = max(self.token_map.keys()) if self.token_map else 0
            mapping_tensor = torch.full((max_token + 1,), self.scaffold_tokenizer.unk_token_id, dtype=torch.long)
            
            # Fill in the mappings
            for base_id, scaffold_id in self.token_map.items():
                mapping_tensor[base_id] = scaffold_id
            
            # Apply the mapping
            return mapping_tensor[base_input_ids]
        except Exception as e:
            logger.log_error(f"Error in map_sequence: {e}")
            raise
    
    def update_token_map_memory(self, prompt: str, confidence: float) -> None:
        """
        Update token mapping memory based on interaction confidence.
        
        Args:
            prompt: Input prompt
            confidence: Confidence score of the interaction
        """
        try:
            if confidence < 0.7:
                return
            
            # Tokenize the prompt with both tokenizers
            base_tokens = self.base_tokenizer.encode(prompt, add_special_tokens=False)
            scaffold_tokens = self.scaffold_tokenizer.encode(prompt, add_special_tokens=False)
            
            # Update the mapping for each token pair
            for base_token, scaffold_token in zip(base_tokens, scaffold_tokens):
                if base_token not in self.token_map or confidence > 0.9:  # Update if new or high confidence
                    self.token_map[base_token] = scaffold_token
        except Exception as e:
            logger.log_error(f"Error in update_token_map_memory: {e}")
            raise
    
    def build_token_map(self) -> None:
        """Build initial token mapping between base and scaffold tokenizers."""
        try:
            # Get vocabulary from both tokenizers
            base_vocab = self.base_tokenizer.get_vocab()
            scaffold_vocab = self.scaffold_tokenizer.get_vocab()
            
            # Build mapping for common tokens
            for token, base_id in base_vocab.items():
                if token in scaffold_vocab:
                    scaffold_id = scaffold_vocab[token]
                    self.token_map[base_id] = scaffold_id
        except Exception as e:
            logger.log_error(f"Error in build_token_map: {e}")
            raise 