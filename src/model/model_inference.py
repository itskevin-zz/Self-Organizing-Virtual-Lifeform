from typing import Dict, Any, Optional, Callable, List, Union
import torch
from torch.utils.data import DataLoader
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .model_config import ModelConfig
from .model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class ModelInference:
    """Handles model inference with extensibility hooks."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        evaluator: Optional[ModelEvaluator] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.evaluator = evaluator or ModelEvaluator(model, config)
        
        # Hooks for future extensibility
        self.pre_inference_hooks: List[Callable] = []
        self.post_inference_hooks: List[Callable] = []
        self.pre_batch_hooks: List[Callable] = []
        self.post_batch_hooks: List[Callable] = []
        
    def predict(
        self,
        inputs: Union[Dict[str, torch.Tensor], DataLoader],
        return_probs: bool = False,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run inference on inputs."""
        # Run pre-inference hooks
        for hook in self.pre_inference_hooks:
            hook(self.model, inputs)
            
        try:
            if isinstance(inputs, DataLoader):
                return self._predict_dataloader(inputs, return_probs)
            else:
                return self._predict_batch(inputs, return_probs)
                
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise
            
    def _predict_dataloader(
        self,
        dataloader: DataLoader,
        return_probs: bool
    ) -> Dict[str, Any]:
        """Run inference on a dataloader."""
        all_predictions = []
        all_probs = [] if return_probs else None
        
        progress_bar = tqdm(dataloader, desc="Inference")
        for batch_idx, batch in enumerate(progress_bar):
            # Run pre-batch hooks
            for hook in self.pre_batch_hooks:
                hook(self.model, batch, batch_idx)
                
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
                    
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                
                if return_probs:
                    probs = torch.softmax(outputs.logits, dim=-1)
                    all_probs.append(probs.cpu().numpy())
                    
            all_predictions.extend(predictions.cpu().numpy())
            
            # Run post-batch hooks
            for hook in self.post_batch_hooks:
                hook(self.model, predictions, batch_idx)
                
        results = {'predictions': np.array(all_predictions)}
        if return_probs:
            results['probabilities'] = np.concatenate(all_probs)
            
        # Run post-inference hooks
        for hook in self.post_inference_hooks:
            hook(self.model, results)
            
        return results
        
    def _predict_batch(
        self,
        inputs: Dict[str, torch.Tensor],
        return_probs: bool
    ) -> Dict[str, Any]:
        """Run inference on a single batch."""
        # Run pre-batch hooks
        for hook in self.pre_batch_hooks:
            hook(self.model, inputs, 0)
            
        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
                 
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
            
            results = {
                'predictions': predictions.cpu().numpy()
            }
            
            if return_probs:
                probs = torch.softmax(outputs.logits, dim=-1)
                results['probabilities'] = probs.cpu().numpy()
                
        # Run post-batch hooks
        for hook in self.post_batch_hooks:
            hook(self.model, predictions, 0)
            
        # Run post-inference hooks
        for hook in self.post_inference_hooks:
            hook(self.model, results)
            
        return results
        
    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """Evaluate model on a dataloader."""
        return self.evaluator.evaluate(dataloader, return_predictions)
        
    def register_pre_inference_hook(self, hook: Callable) -> None:
        """Register a hook to run before inference."""
        self.pre_inference_hooks.append(hook)
        
    def register_post_inference_hook(self, hook: Callable) -> None:
        """Register a hook to run after inference."""
        self.post_inference_hooks.append(hook)
        
    def register_pre_batch_hook(self, hook: Callable) -> None:
        """Register a hook to run before each batch."""
        self.pre_batch_hooks.append(hook)
        
    def register_post_batch_hook(self, hook: Callable) -> None:
        """Register a hook to run after each batch."""
        self.post_batch_hooks.append(hook) 