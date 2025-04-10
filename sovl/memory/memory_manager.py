from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
import time
import math
from collections import defaultdict

class MemoryManager:
    """
    Manages the memory system for the model.
    """
    def __init__(
        self,
        max_token_map_size: int = 10000,
        max_scaffold_memory: int = 1000,
        token_map_decay: float = 0.99,
        scaffold_memory_decay: float = 0.95,
        min_token_frequency: int = 3
    ):
        self.token_map = defaultdict(int)
        self.scaffold_memory = []
        self.max_token_map_size = max_token_map_size
        self.max_scaffold_memory = max_scaffold_memory
        self.token_map_decay = token_map_decay
        self.scaffold_memory_decay = scaffold_memory_decay
        self.min_token_frequency = min_token_frequency
        self.last_decay_time = time.time()

    def update_token_map(
        self,
        token_ids: torch.Tensor,
        frequencies: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update the token map with new token frequencies.
        
        Args:
            token_ids: Tensor of token IDs
            frequencies: Optional tensor of token frequencies
        """
        if frequencies is None:
            frequencies = torch.ones_like(token_ids)
        
        for token_id, freq in zip(token_ids.tolist(), frequencies.tolist()):
            self.token_map[token_id] += freq

    def decay_token_map(self) -> None:
        """
        Apply decay to token frequencies.
        """
        for token_id in list(self.token_map.keys()):
            self.token_map[token_id] *= self.token_map_decay
            if self.token_map[token_id] < self.min_token_frequency:
                del self.token_map[token_id]

    def prune_token_map(self) -> None:
        """
        Prune the token map to maintain size limits.
        """
        if len(self.token_map) > self.max_token_map_size:
            # Sort by frequency and keep top tokens
            sorted_tokens = sorted(
                self.token_map.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.token_map = dict(sorted_tokens[:self.max_token_map_size])

    def add_scaffold_memory(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Add new scaffold memory.
        
        Args:
            hidden_states: Hidden states from scaffold model
            attention_mask: Attention mask
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        memory_entry = {
            "hidden_states": hidden_states.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "timestamp": timestamp,
            "weight": 1.0
        }
        
        self.scaffold_memory.append(memory_entry)
        
        # Maintain size limit
        if len(self.scaffold_memory) > self.max_scaffold_memory:
            self.scaffold_memory.pop(0)

    def decay_scaffold_memory(self) -> None:
        """
        Apply decay to scaffold memory weights.
        """
        for entry in self.scaffold_memory:
            entry["weight"] *= self.scaffold_memory_decay

    def get_scaffold_context(
        self,
        model: PreTrainedModel,
        max_memories: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get scaffold context from memory.
        
        Args:
            model: Model to use
            max_memories: Maximum number of memories to use
            
        Returns:
            Tuple of (hidden states, attention mask)
        """
        if not self.scaffold_memory:
            return None, None
        
        # Select memories based on weights and recency
        weights = [entry["weight"] for entry in self.scaffold_memory]
        selected_idx = sorted(
            range(len(self.scaffold_memory)),
            key=lambda i: weights[i],
            reverse=True
        )[:max_memories]
        
        # Combine selected memories
        hidden_states = []
        attention_masks = []
        
        for idx in selected_idx:
            entry = self.scaffold_memory[idx]
            hidden_states.append(entry["hidden_states"])
            attention_masks.append(entry["attention_mask"])
        
        # Stack tensors
        hidden_states = torch.stack(hidden_states).to(model.device)
        attention_masks = torch.stack(attention_masks).to(model.device)
        
        return hidden_states, attention_masks

    def update(self) -> None:
        """
        Update memory system (decay and pruning).
        """
        current_time = time.time()
        time_since_last_decay = current_time - self.last_decay_time
        
        # Apply decay periodically
        if time_since_last_decay >= 3600:  # 1 hour
            self.decay_token_map()
            self.decay_scaffold_memory()
            self.prune_token_map()
            self.last_decay_time = current_time

    def get_token_weights(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get weights for tokens based on their frequencies.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Tensor of token weights
        """
        weights = torch.zeros_like(token_ids, dtype=torch.float)
        for i, token_id in enumerate(token_ids.tolist()):
            weights[i] = self.token_map.get(token_id, 1.0)
        
        return weights

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        return {
            "token_map_size": len(self.token_map),
            "scaffold_memory_size": len(self.scaffold_memory),
            "avg_token_frequency": sum(self.token_map.values()) / len(self.token_map) if self.token_map else 0.0,
            "avg_scaffold_weight": sum(entry["weight"] for entry in self.scaffold_memory) / len(self.scaffold_memory) if self.scaffold_memory else 0.0
        } 