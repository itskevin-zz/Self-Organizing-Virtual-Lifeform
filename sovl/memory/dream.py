from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
import random
import time
import math
from collections import deque

class DreamMemory:
    """
    Manages the dream memory system.
    """
    def __init__(
        self,
        max_size: int = 100,
        decay_rate: float = 0.95,
        novelty_boost: float = 0.1,
        memory_weight: float = 0.5
    ):
        self.memory = deque(maxlen=max_size)
        self.decay_rate = decay_rate
        self.novelty_boost = novelty_boost
        self.memory_weight = memory_weight
        self.last_dream_time = time.time()

    def add_memory(
        self,
        prompt: str,
        response: str,
        confidence: float,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Add a new memory to the dream memory.
        
        Args:
            prompt: Input prompt
            response: Model response
            confidence: Confidence score
            timestamp: Timestamp (optional)
        """
        if timestamp is None:
            timestamp = time.time()
        
        memory_entry = {
            "prompt": prompt,
            "response": response,
            "confidence": confidence,
            "timestamp": timestamp,
            "weight": 1.0
        }
        
        self.memory.append(memory_entry)

    def decay_memory(self) -> None:
        """
        Apply decay to all memories.
        """
        for entry in self.memory:
            entry["weight"] *= self.decay_rate

    def calculate_novelty(self, prompt: str) -> float:
        """
        Calculate novelty score for a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Novelty score between 0 and 1
        """
        if not self.memory:
            return 1.0
        
        # Calculate similarity with existing memories
        similarities = []
        for entry in self.memory:
            # Simple character-level similarity
            similarity = self._calculate_similarity(prompt, entry["prompt"])
            similarities.append(similarity)
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        return 1.0 - max_similarity

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate similarity between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple character-level similarity
        common_chars = set(s1) & set(s2)
        total_chars = set(s1) | set(s2)
        return len(common_chars) / len(total_chars) if total_chars else 0.0

    def get_dream_context(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ) -> Tuple[torch.Tensor, float]:
        """
        Get context for dreaming from memory.
        
        Args:
            model: Model to use
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (context tensor, novelty score)
        """
        if not self.memory:
            return None, 1.0
        
        # Select memories based on weights and recency
        weights = [entry["weight"] for entry in self.memory]
        selected_idx = random.choices(
            range(len(self.memory)),
            weights=weights,
            k=min(3, len(self.memory))
        )
        
        # Combine selected memories
        context_parts = []
        for idx in selected_idx:
            entry = self.memory[idx]
            context_parts.append(f"Prompt: {entry['prompt']}\nResponse: {entry['response']}\n")
        
        context = "\n".join(context_parts)
        
        # Tokenize and truncate
        inputs = tokenizer(
            context,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Calculate novelty
        novelty = self.calculate_novelty(context)
        
        return inputs["input_ids"].to(model.device), novelty

    def should_dream(
        self,
        time_since_last_dream: float,
        novelty_threshold: float = 0.3,
        time_factor: float = 3600.0
    ) -> bool:
        """
        Determine if dreaming should occur.
        
        Args:
            time_since_last_dream: Time since last dream
            novelty_threshold: Novelty threshold
            time_factor: Time scaling factor
            
        Returns:
            True if dreaming should occur, False otherwise
        """
        # Calculate time-based probability
        time_prob = 1.0 - math.exp(-time_since_last_dream / time_factor)
        
        # Calculate novelty-based probability
        if self.memory:
            avg_novelty = sum(
                self.calculate_novelty(entry["prompt"])
                for entry in self.memory
            ) / len(self.memory)
            novelty_prob = max(0.0, avg_novelty - novelty_threshold) / (1.0 - novelty_threshold)
        else:
            novelty_prob = 1.0
        
        # Combined probability
        prob = time_prob * novelty_prob
        
        return random.random() < prob

    def dream(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> Dict[str, Any]:
        """
        Perform dreaming.
        
        Args:
            model: Model to use
            tokenizer: Tokenizer to use
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary containing dream results
        """
        # Get dream context
        context, novelty = self.get_dream_context(model, tokenizer)
        
        # Generate dream
        with torch.no_grad():
            outputs = model.generate(
                context,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True
            )
        
        # Decode dream
        dream_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Update last dream time
        self.last_dream_time = time.time()
        
        return {
            "dream_text": dream_text,
            "novelty": novelty,
            "timestamp": self.last_dream_time
        } 