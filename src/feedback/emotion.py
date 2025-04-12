from typing import Dict, List, Optional, Tuple
import torch
import time
from collections import deque
from ..core.system_config import config_manager
from ..core.system_logging import logger
from ..models.model_loader import model_loader

class EmotionalState:
    """Manages the emotional state of the system."""
    
    def __init__(self):
        self.valence = 0.0  # Positive vs negative (-1 to 1)
        self.arousal = 0.0  # Energy level (0 to 1)
        self.dominance = 0.5  # Feeling of control (0 to 1)
        self.history = deque(maxlen=100)
        self.last_update = time.time()
        self.scaffold_model = model_loader.scaffold_model
    
    def update_state(
        self,
        interaction: str,
        success: bool = True,
        confidence: float = 0.5,
        external_feedback: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update emotional state based on interaction outcome.
        
        Args:
            interaction: The interaction text
            success: Whether the interaction was successful
            confidence: Confidence level in the interaction
            external_feedback: Optional external emotional feedback
        """
        try:
            # Get base emotion weights
            controls_config = config_manager.get_value("controls_config")
            success_impact = controls_config.get("emotion_success_impact", 0.3)
            confidence_impact = controls_config.get("emotion_confidence_impact", 0.2)
            
            # Calculate sentiment of interaction
            sentiment = self._analyze_sentiment(interaction)
            
            # Update valence (positive/negative)
            valence_change = (
                0.2 * sentiment +                          # Text sentiment
                success_impact * (1.0 if success else -1.0) +  # Success/failure
                confidence_impact * (confidence - 0.5)     # Confidence impact
            )
            self.valence = max(-1.0, min(1.0, self.valence + valence_change))
            
            # Update arousal (energy)
            arousal_change = (
                0.1 * abs(sentiment) +                     # Emotional intensity
                0.2 * (1.0 if success else 0.5) +         # Success energizes
                0.1 * confidence                          # Confidence energizes
            )
            self.arousal = max(0.0, min(1.0, self.arousal + arousal_change))
            
            # Update dominance (control)
            dominance_change = (
                0.2 * (1.0 if success else -0.5) +        # Success increases control
                0.3 * (confidence - 0.5)                  # Confidence affects control
            )
            self.dominance = max(0.0, min(1.0, self.dominance + dominance_change))
            
            # Apply external feedback if provided
            if external_feedback:
                self._apply_external_feedback(external_feedback)
            
            # Record state
            self.history.append({
                'valence': self.valence,
                'arousal': self.arousal,
                'dominance': self.dominance,
                'timestamp': time.time()
            })
            
            self.last_update = time.time()
            
        except Exception as e:
            logger.log_error(f"Error updating emotional state: {e}")
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using the model.
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Sentiment score between -1 and 1
        """
        try:
            # Tokenize text
            inputs = model_loader.scaffold_tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Get model's sentiment prediction
            with torch.no_grad():
                outputs = self.scaffold_model(**inputs)
                logits = outputs.logits
                
                # Convert logits to sentiment score
                sentiment = torch.tanh(logits.mean())
                
                return sentiment.item()
                
        except Exception as e:
            logger.log_error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def _apply_external_feedback(self, feedback: Dict[str, float]) -> None:
        """
        Apply external emotional feedback.
        
        Args:
            feedback: Dictionary with emotional feedback values
        """
        try:
            if 'valence' in feedback:
                self.valence = max(-1.0, min(1.0, 
                    0.7 * self.valence + 0.3 * feedback['valence']
                ))
            
            if 'arousal' in feedback:
                self.arousal = max(0.0, min(1.0,
                    0.7 * self.arousal + 0.3 * feedback['arousal']
                ))
                
            if 'dominance' in feedback:
                self.dominance = max(0.0, min(1.0,
                    0.7 * self.dominance + 0.3 * feedback['dominance']
                ))
                
        except Exception as e:
            logger.log_error(f"Error applying external feedback: {e}")
    
    def get_current_state(self) -> Dict[str, float]:
        """
        Get current emotional state.
        
        Returns:
            Dict containing current emotional values
        """
        return {
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance
        }
    
    def get_mood(self) -> str:
        """
        Get current mood based on emotional state.
        
        Returns:
            str: Description of current mood
        """
        try:
            # Define mood thresholds
            if self.valence > 0.5:
                if self.arousal > 0.7:
                    return "excited" if self.dominance > 0.5 else "enthusiastic"
                elif self.arousal > 0.3:
                    return "happy" if self.dominance > 0.5 else "content"
                else:
                    return "calm" if self.dominance > 0.5 else "relaxed"
            elif self.valence < -0.5:
                if self.arousal > 0.7:
                    return "angry" if self.dominance > 0.5 else "frustrated"
                elif self.arousal > 0.3:
                    return "sad" if self.dominance > 0.5 else "disappointed"
                else:
                    return "depressed" if self.dominance > 0.5 else "exhausted"
            else:
                if self.arousal > 0.7:
                    return "alert" if self.dominance > 0.5 else "tense"
                elif self.arousal > 0.3:
                    return "neutral" if self.dominance > 0.5 else "uncertain"
                else:
                    return "bored" if self.dominance > 0.5 else "tired"
                    
        except Exception as e:
            logger.log_error(f"Error getting mood: {e}")
            return "neutral"

# Create singleton instance
emotional_state = EmotionalState() 