from typing import Dict, Any, Optional, Callable, List
import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from tqdm import tqdm
import time
from .model_config import ModelConfig
from .model_optimizer import ModelOptimizer
from .model_evaluator import ModelEvaluator
from .model_saver import ModelSaver

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training with extensibility hooks."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[ModelOptimizer] = None,
        evaluator: Optional[ModelEvaluator] = None,
        saver: Optional[ModelSaver] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.model.to(device)
        
        # Initialize components
        self.optimizer = optimizer or ModelOptimizer(model, config)
        self.evaluator = evaluator or ModelEvaluator(model, config)
        self.saver = saver or ModelSaver(model, config, "checkpoints")
        
        # Training state
        self.current_epoch = 0
        self.best_metrics: Dict[str, float] = {}
        self.training_history: List[Dict[str, Any]] = []
        
        # Hooks for future extensibility
        self.pre_epoch_hooks: List[Callable] = []
        self.post_epoch_hooks: List[Callable] = []
        self.pre_batch_hooks: List[Callable] = []
        self.post_batch_hooks: List[Callable] = []
        self.validation_hooks: List[Callable] = []
        
    def train(
        self,
        num_epochs: int,
        save_every: int = 1,
        validate_every: int = 1,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train the model for specified number of epochs."""
        start_time = time.time()
        best_epoch = 0
        patience_counter = 0
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # Run pre-epoch hooks
                for hook in self.pre_epoch_hooks:
                    hook(self.model, epoch)
                    
                # Training loop
                train_metrics = self._train_epoch()
                
                # Validation if needed
                val_metrics = {}
                if self.val_dataloader and (epoch + 1) % validate_every == 0:
                    val_metrics = self._validate_epoch()
                    
                # Update training history
                epoch_metrics = {
                    'epoch': epoch,
                    'train': train_metrics,
                    'val': val_metrics,
                    'time': time.time() - start_time
                }
                self.training_history.append(epoch_metrics)
                
                # Save checkpoint if needed
                if (epoch + 1) % save_every == 0:
                    self._save_checkpoint(epoch_metrics)
                    
                # Early stopping check
                if early_stopping_patience and self.val_dataloader:
                    if self._should_stop_early(val_metrics, patience_counter):
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
                        
                # Run post-epoch hooks
                for hook in self.post_epoch_hooks:
                    hook(self.model, epoch_metrics)
                    
            return {
                'training_history': self.training_history,
                'best_metrics': self.best_metrics,
                'best_epoch': best_epoch
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {}
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            # Run pre-batch hooks
            for hook in self.pre_batch_hooks:
                hook(self.model, batch, batch_idx)
                
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
                    
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Optimization step
            self.optimizer.step(loss)
            
            # Update metrics
            batch_metrics = {
                'loss': loss.item(),
                'lr': self.optimizer.get_learning_rate()
            }
            
            # Run post-batch hooks
            for hook in self.post_batch_hooks:
                hook(self.model, batch_metrics, batch_idx)
                
            # Update progress bar
            progress_bar.set_postfix(batch_metrics)
            
        return batch_metrics
        
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if not self.val_dataloader:
            return {}
            
        # Run validation hooks
        for hook in self.validation_hooks:
            hook(self.model, self.val_dataloader)
            
        return self.evaluator.evaluate(self.val_dataloader)
        
    def _save_checkpoint(self, metrics: Dict[str, Any]) -> None:
        """Save training checkpoint."""
        is_best = False
        if self.val_dataloader and metrics['val']:
            val_metrics = metrics['val']
            if not self.best_metrics or val_metrics['loss'] < self.best_metrics.get('loss', float('inf')):
                self.best_metrics = val_metrics
                is_best = True
                
        self.saver.save_checkpoint(
            name=f"epoch_{self.current_epoch}",
            epoch=self.current_epoch,
            metrics=metrics,
            is_best=is_best
        )
        
    def _should_stop_early(
        self,
        val_metrics: Dict[str, float],
        patience_counter: int
    ) -> bool:
        """Check if training should stop early."""
        if not self.best_metrics:
            return False
            
        if val_metrics['loss'] < self.best_metrics['loss']:
            return False
            
        patience_counter += 1
        return patience_counter >= self.early_stopping_patience
        
    def register_pre_epoch_hook(self, hook: Callable) -> None:
        """Register a hook to run before each epoch."""
        self.pre_epoch_hooks.append(hook)
        
    def register_post_epoch_hook(self, hook: Callable) -> None:
        """Register a hook to run after each epoch."""
        self.post_epoch_hooks.append(hook)
        
    def register_pre_batch_hook(self, hook: Callable) -> None:
        """Register a hook to run before each batch."""
        self.pre_batch_hooks.append(hook)
        
    def register_post_batch_hook(self, hook: Callable) -> None:
        """Register a hook to run after each batch."""
        self.post_batch_hooks.append(hook)
        
    def register_validation_hook(self, hook: Callable) -> None:
        """Register a hook for validation."""
        self.validation_hooks.append(hook)
        
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history
        
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best validation metrics."""
        return self.best_metrics 