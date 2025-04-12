from typing import Dict, List, Optional, Tuple
import torch
import random
import time
from collections import deque
from ..core.system_config import config_manager
from ..core.system_logging import logger
from ..models.model_loader import model_loader

class DreamManager:
    """Manages the dreaming process for model adaptation."""
    
    def __init__(self):
        self.last_dream_time = 0
        self.is_dreaming = False
        self.dream_memory: deque = deque(maxlen=100)  # Store recent dreams
        self.scaffold_model = model_loader.scaffold_model
        self.base_model = model_loader.base_model
    
    def should_dream(self) -> bool:
        """
        Check if dreaming should occur based on configuration and conditions.
        
        Returns:
            bool indicating whether dreaming should occur
        """
        try:
            controls_config = config_manager.get_value("controls_config")
            
            # Check if dreaming is enabled
            if not controls_config.get("enable_dreaming", True):
                return False
            
            # Check time since last dream
            current_time = time.time()
            time_since_last = current_time - self.last_dream_time
            min_interval = controls_config.get("dream_interval", 600)  # 10 minutes default
            
            # Check dream conditions
            dream_swing_var = controls_config.get("dream_swing_var", 0.1)
            dream_lifecycle_delta = controls_config.get("dream_lifecycle_delta", 0.1)
            
            # Additional conditions can be added here
            
            return time_since_last >= min_interval
            
        except Exception as e:
            logger.log_error(f"Error checking dream condition: {e}")
            return False
    
    def dream(self) -> Dict[str, float]:
        """
        Perform the dreaming process.
        
        Returns:
            Dictionary containing dream metrics
        """
        try:
            if self.is_dreaming:
                logger.log_info("Dream already in progress")
                return {}
            
            self.is_dreaming = True
            self.last_dream_time = time.time()
            
            # Get configuration
            controls_config = config_manager.get_value("controls_config")
            
            # Generate dream content
            dream_content = self._generate_dream_content()
            if not dream_content:
                self.is_dreaming = False
                return {}
            
            # Process dream
            metrics = self._process_dream(dream_content)
            
            # Update dream memory
            self._update_dream_memory(dream_content, metrics)
            
            self.is_dreaming = False
            return metrics
            
        except Exception as e:
            self.is_dreaming = False
            logger.log_error(f"Error during dreaming: {e}")
            raise
    
    def _generate_dream_content(self) -> Optional[Dict[str, torch.Tensor]]:
        """Generate content for dreaming."""
        try:
            controls_config = config_manager.get_value("controls_config")
            
            # Get recent interactions from memory
            interactions = list(self.dream_memory)
            if not interactions:
                return None
            
            # Select random interaction as base
            base_interaction = random.choice(interactions)
            
            # Apply dream variations
            noise_scale = controls_config.get("dream_noise_scale", 0.05)
            dream_prompt = base_interaction["prompt"]
            
            # Tokenize dream content
            inputs = model_loader.scaffold_tokenizer(
                dream_prompt,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "original_prompt": dream_prompt
            }
            
        except Exception as e:
            logger.log_error(f"Error generating dream content: {e}")
            return None
    
    def _process_dream(self, dream_content: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Process the dream content and generate metrics."""
        try:
            controls_config = config_manager.get_value("controls_config")
            
            # Generate response from both models
            with torch.no_grad():
                base_output = self.base_model(
                    input_ids=dream_content["input_ids"],
                    attention_mask=dream_content["attention_mask"]
                )
                scaffold_output = self.scaffold_model(
                    input_ids=dream_content["input_ids"],
                    attention_mask=dream_content["attention_mask"]
                )
            
            # Calculate dream metrics
            novelty = self._calculate_novelty(base_output.logits, scaffold_output.logits)
            coherence = self._calculate_coherence(scaffold_output.logits)
            
            return {
                "novelty": novelty,
                "coherence": coherence
            }
            
        except Exception as e:
            logger.log_error(f"Error processing dream: {e}")
            return {}
    
    def _calculate_novelty(self, base_logits: torch.Tensor, scaffold_logits: torch.Tensor) -> float:
        """Calculate novelty score between base and scaffold outputs."""
        try:
            # Calculate KL divergence or other similarity metric
            base_probs = torch.softmax(base_logits, dim=-1)
            scaffold_probs = torch.softmax(scaffold_logits, dim=-1)
            
            kl_div = torch.nn.functional.kl_div(
                base_probs.log(),
                scaffold_probs,
                reduction='batchmean'
            )
            
            return kl_div.item()
            
        except Exception as e:
            logger.log_error(f"Error calculating novelty: {e}")
            return 0.0
    
    def _calculate_coherence(self, logits: torch.Tensor) -> float:
        """Calculate coherence score of the output."""
        try:
            # Calculate entropy or other coherence metric
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
            
            # Normalize to [0, 1]
            coherence = 1.0 - torch.clamp(entropy / 10.0, 0.0, 1.0)
            
            return coherence.item()
            
        except Exception as e:
            logger.log_error(f"Error calculating coherence: {e}")
            return 0.0
    
    def _update_dream_memory(self, dream_content: Dict[str, torch.Tensor], metrics: Dict[str, float]) -> None:
        """Update dream memory with new dream content and metrics."""
        try:
            controls_config = config_manager.get_value("controls_config")
            
            # Create memory entry
            memory_entry = {
                "prompt": dream_content["original_prompt"],
                "metrics": metrics,
                "timestamp": time.time()
            }
            
            # Apply memory decay
            decay_rate = controls_config.get("dream_memory_decay", 0.95)
            prune_threshold = controls_config.get("dream_prune_threshold", 0.1)
            
            # Remove old entries below threshold
            self.dream_memory = deque(
                [entry for entry in self.dream_memory if entry["metrics"]["coherence"] > prune_threshold],
                maxlen=self.dream_memory.maxlen
            )
            
            # Add new entry
            self.dream_memory.append(memory_entry)
            
        except Exception as e:
            logger.log_error(f"Error updating dream memory: {e}")

# Create a singleton instance
dream_manager = DreamManager() 