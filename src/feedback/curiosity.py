from typing import Dict, List, Optional, Tuple
import torch
import time
from collections import deque
from ..core.system_config import config_manager
from ..core.system_logging import logger
from ..models.model_loader import model_loader

class TrueCuriosity:
    """Manages model's curiosity mechanism."""
    
    def __init__(self):
        self.question_history: deque = deque(maxlen=100)
        self.last_question_time = 0
        self.pressure = CuriosityPressure()
        self.scaffold_model = model_loader.scaffold_model
    
    def calculate_metric(self, question: str) -> float:
        """
        Calculate curiosity metric for a question.
        
        Args:
            question: The question to evaluate
            
        Returns:
            float: Curiosity score between 0 and 1
        """
        try:
            controls_config = config_manager.get_value("controls_config")
            
            # Get weights for different components
            weight_ignorance = controls_config.get("curiosity_weight_ignorance", 0.7)
            weight_novelty = controls_config.get("curiosity_weight_novelty", 0.3)
            
            # Calculate ignorance score (how much we don't know about this topic)
            ignorance_score = self._calculate_ignorance(question)
            
            # Calculate novelty score (how different this is from previous questions)
            novelty_score = self._calculate_novelty(question)
            
            # Combine scores
            curiosity_score = (
                weight_ignorance * ignorance_score +
                weight_novelty * novelty_score
            )
            
            return float(curiosity_score)
            
        except Exception as e:
            logger.log_error(f"Error calculating curiosity metric: {e}")
            return 0.0
    
    def _calculate_ignorance(self, question: str) -> float:
        """Calculate how much we don't know about the topic."""
        try:
            # Tokenize question
            inputs = model_loader.scaffold_tokenizer(
                question,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Get model's confidence on this input
            with torch.no_grad():
                outputs = self.scaffold_model(**inputs)
                logits = outputs.logits
                
                # Calculate entropy of the predictions
                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                
                # Normalize to [0, 1] where higher means more ignorance
                ignorance = torch.clamp(entropy / 10.0, 0.0, 1.0)
                
                return ignorance.item()
                
        except Exception as e:
            logger.log_error(f"Error calculating ignorance: {e}")
            return 0.0
    
    def _calculate_novelty(self, question: str) -> float:
        """Calculate how novel the question is compared to history."""
        try:
            if not self.question_history:
                return 1.0
            
            # Tokenize current and historical questions
            current_tokens = set(model_loader.scaffold_tokenizer.tokenize(question))
            
            # Calculate average similarity with history
            similarities = []
            for past_question in self.question_history:
                past_tokens = set(model_loader.scaffold_tokenizer.tokenize(past_question))
                
                # Use Jaccard similarity
                intersection = len(current_tokens & past_tokens)
                union = len(current_tokens | past_tokens)
                similarity = intersection / union if union > 0 else 0
                
                similarities.append(similarity)
            
            # Convert similarity to novelty (1 - avg_similarity)
            avg_similarity = sum(similarities) / len(similarities)
            novelty = 1.0 - avg_similarity
            
            return novelty
            
        except Exception as e:
            logger.log_error(f"Error calculating novelty: {e}")
            return 0.0
    
    def update_metrics(
        self,
        question: str,
        score: float,
        spontaneous: bool = False,
        answered: bool = False
    ) -> None:
        """
        Update curiosity metrics after asking a question.
        
        Args:
            question: The question asked
            score: Curiosity score for the question
            spontaneous: Whether the question was spontaneously generated
            answered: Whether the question was answered
        """
        try:
            # Update question history
            self.question_history.append(question)
            
            # Update timing
            self.last_question_time = time.time()
            
            # Update pressure based on outcome
            if answered:
                self.pressure.update(
                    temperament=score,
                    confidence=1.0 if answered else 0.0,
                    silence=0.0
                )
            
        except Exception as e:
            logger.log_error(f"Error updating curiosity metrics: {e}")
    
    def check_silence(self, elapsed: float) -> None:
        """
        Update curiosity based on silence duration.
        
        Args:
            elapsed: Time elapsed since last interaction
        """
        try:
            controls_config = config_manager.get_value("controls_config")
            silence_threshold = controls_config.get("curiosity_silence_threshold", 20.0)
            
            if elapsed > silence_threshold:
                # Increase pressure with silence
                silence_factor = min(elapsed / silence_threshold, 2.0)
                self.pressure.update(
                    temperament=0.5,  # Neutral temperament during silence
                    confidence=0.0,   # No confidence during silence
                    silence=silence_factor
                )
            
        except Exception as e:
            logger.log_error(f"Error checking silence: {e}")

class CuriosityPressure:
    """Manages the build-up and release of curiosity pressure."""
    
    def __init__(self):
        self.pressure = 0.0
        self.last_update = time.time()
    
    def update(self, temperament: float, confidence: float, silence: float) -> None:
        """
        Update curiosity pressure based on various factors.
        
        Args:
            temperament: Current temperament value
            confidence: Confidence in recent interactions
            silence: Factor representing duration of silence
        """
        try:
            controls_config = config_manager.get_value("controls_config")
            
            # Get pressure parameters
            pressure_threshold = controls_config.get("curiosity_pressure_threshold", 0.7)
            pressure_drop = controls_config.get("curiosity_pressure_drop", 0.3)
            
            # Calculate time factor
            current_time = time.time()
            time_delta = current_time - self.last_update
            time_factor = min(time_delta / 3600.0, 1.0)  # Cap at 1 hour
            
            # Update pressure based on factors
            if confidence > 0.8:  # High confidence reduces pressure
                self.pressure = max(0.0, self.pressure - pressure_drop * time_factor)
            else:
                # Increase pressure based on temperament and silence
                pressure_increase = (
                    0.1 * temperament +    # Higher temperament = faster increase
                    0.2 * silence +        # Longer silence = more pressure
                    0.05 * time_factor     # Natural increase over time
                )
                self.pressure = min(1.0, self.pressure + pressure_increase)
            
            self.last_update = current_time
            
        except Exception as e:
            logger.log_error(f"Error updating curiosity pressure: {e}")
    
    def should_erupt(self, threshold: Optional[float] = None) -> bool:
        """
        Check if curiosity pressure should cause a question.
        
        Args:
            threshold: Optional override for pressure threshold
            
        Returns:
            bool indicating whether pressure should be released
        """
        try:
            if threshold is None:
                controls_config = config_manager.get_value("controls_config")
                threshold = controls_config.get("curiosity_pressure_threshold", 0.7)
            
            return self.pressure >= threshold
            
        except Exception as e:
            logger.log_error(f"Error checking curiosity eruption: {e}")
            return False

# Create singleton instances
curiosity = TrueCuriosity() 