from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
from collections import deque

class CuriosityModule:
    """
    Manages curiosity and feedback mechanisms for the model.
    """
    def __init__(
        self,
        max_history: int = 1000,
        novelty_threshold: float = 0.7,
        exploration_rate: float = 0.1,
        feedback_weight: float = 0.5
    ):
        self.history = deque(maxlen=max_history)
        self.novelty_threshold = novelty_threshold
        self.exploration_rate = exploration_rate
        self.feedback_weight = feedback_weight
        self.last_feedback = None

    def calculate_novelty(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> float:
        """
        Calculate novelty score for given hidden states.
        
        Args:
            hidden_states: Hidden states from model
            attention_mask: Attention mask
            
        Returns:
            Novelty score between 0 and 1
        """
        if not self.history:
            return 1.0
        
        # Calculate mean hidden states
        mean_states = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]
        
        # Compare with history
        similarities = []
        for past_states in self.history:
            past_mean = torch.mean(past_states, dim=1)
            similarity = F.cosine_similarity(mean_states, past_mean, dim=1)
            similarities.append(similarity)
        
        # Get minimum similarity (maximum novelty)
        min_similarity = torch.min(torch.stack(similarities))
        novelty = 1.0 - min_similarity
        
        return novelty.item()

    def should_explore(self, novelty: float) -> bool:
        """
        Determine if model should explore based on novelty.
        
        Args:
            novelty: Novelty score
            
        Returns:
            Whether to explore
        """
        return novelty > self.novelty_threshold or np.random.random() < self.exploration_rate

    def add_to_history(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> None:
        """
        Add hidden states to history.
        
        Args:
            hidden_states: Hidden states to add
            attention_mask: Attention mask
        """
        self.history.append(hidden_states.detach().cpu())

    def update_feedback(
        self,
        feedback: Dict[str, Any]
    ) -> None:
        """
        Update feedback from environment.
        
        Args:
            feedback: Dictionary containing feedback information
        """
        self.last_feedback = feedback

    def get_feedback_weights(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Get weights based on feedback.
        
        Args:
            hidden_states: Hidden states to weight
            
        Returns:
            Tensor of weights
        """
        if self.last_feedback is None:
            return torch.ones_like(hidden_states)
        
        # Calculate feedback-based weights
        weights = torch.ones_like(hidden_states)
        
        # Apply feedback based on last feedback
        if "positive" in self.last_feedback:
            weights *= (1.0 + self.feedback_weight)
        elif "negative" in self.last_feedback:
            weights *= (1.0 - self.feedback_weight)
        
        return weights

    def get_curiosity_stats(self) -> Dict[str, Any]:
        """
        Get curiosity module statistics.
        
        Returns:
            Dictionary containing curiosity statistics
        """
        return {
            "history_size": len(self.history),
            "last_feedback": self.last_feedback,
            "exploration_rate": self.exploration_rate
        } 