from typing import Dict, Any, Optional, Callable, List, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
import logging
from .model_config import ModelConfig

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation with extensibility hooks."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        metrics: Optional[Dict[str, Callable]] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Default metrics if none provided
        self.metrics = metrics or {
            'accuracy': self._calculate_accuracy,
            'loss': self._calculate_loss
        }
        
        # Hooks for future extensibility
        self.pre_evaluation_hooks: List[Callable] = []
        self.post_evaluation_hooks: List[Callable] = []
        self.metric_hooks: List[Callable] = []
        
    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """Evaluate model on given dataloader."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        # Run pre-evaluation hooks
        for hook in self.pre_evaluation_hooks:
            hook(self.model, dataloader)
            
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Store predictions and targets
                if return_predictions:
                    predictions = outputs.logits.argmax(dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(batch['labels'].cpu().numpy())
                
                # Calculate loss
                total_loss += outputs.loss.item()
                
        # Calculate metrics
        results = {}
        for metric_name, metric_fn in self.metrics.items():
            try:
                if metric_name == 'loss':
                    results[metric_name] = total_loss / len(dataloader)
                else:
                    results[metric_name] = metric_fn(
                        np.array(all_predictions),
                        np.array(all_targets)
                    )
            except Exception as e:
                logger.error(f"Error calculating metric {metric_name}: {str(e)}")
                results[metric_name] = float('nan')
                
        # Run metric hooks
        for hook in self.metric_hooks:
            hook(results)
            
        # Run post-evaluation hooks
        for hook in self.post_evaluation_hooks:
            hook(self.model, results)
            
        if return_predictions:
            return results, (all_predictions, all_targets)
        return results
        
    def _calculate_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate accuracy metric."""
        return np.mean(predictions == targets)
        
    def _calculate_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate loss metric."""
        return np.mean((predictions - targets) ** 2)
        
    def register_pre_evaluation_hook(self, hook: Callable) -> None:
        """Register a hook to run before evaluation."""
        self.pre_evaluation_hooks.append(hook)
        
    def register_post_evaluation_hook(self, hook: Callable) -> None:
        """Register a hook to run after evaluation."""
        self.post_evaluation_hooks.append(hook)
        
    def register_metric_hook(self, hook: Callable) -> None:
        """Register a hook for custom metric calculation."""
        self.metric_hooks.append(hook)
        
    def add_metric(self, name: str, metric_fn: Callable) -> None:
        """Add a new metric to the evaluator."""
        self.metrics[name] = metric_fn
        
    def get_metric_names(self) -> List[str]:
        """Get list of available metric names."""
        return list(self.metrics.keys()) 