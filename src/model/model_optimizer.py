from typing import Dict, Any, Optional, Callable
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import logging
from .model_config import ModelConfig

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Handles model optimization with extensibility hooks."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        optimizer_class: type[Optimizer] = torch.optim.AdamW,
        scheduler_class: Optional[type[LRScheduler]] = None,
        custom_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        custom_scheduler_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.config = config
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.custom_optimizer_kwargs = custom_optimizer_kwargs or {}
        self.custom_scheduler_kwargs = custom_scheduler_kwargs or {}
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler() if scheduler_class else None
        
        # Hooks for future extensibility
        self.pre_optimization_hooks: list[Callable] = []
        self.post_optimization_hooks: list[Callable] = []
        self.gradient_clipping_hooks: list[Callable] = []
        
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer with default or custom parameters."""
        default_kwargs = {
            "lr": 1e-4,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8
        }
        
        # Merge default and custom kwargs
        optimizer_kwargs = {**default_kwargs, **self.custom_optimizer_kwargs}
        
        try:
            return self.optimizer_class(self.model.parameters(), **optimizer_kwargs)
        except Exception as e:
            logger.error(f"Failed to create optimizer: {str(e)}")
            raise
            
    def _create_scheduler(self) -> Optional[LRScheduler]:
        """Create learning rate scheduler if specified."""
        if not self.scheduler_class:
            return None
            
        default_kwargs = {
            "optimizer": self.optimizer,
            "num_warmup_steps": 100,
            "num_training_steps": 1000
        }
        
        # Merge default and custom kwargs
        scheduler_kwargs = {**default_kwargs, **self.custom_scheduler_kwargs}
        
        try:
            return self.scheduler_class(**scheduler_kwargs)
        except Exception as e:
            logger.error(f"Failed to create scheduler: {str(e)}")
            raise
            
    def step(self, loss: torch.Tensor) -> None:
        """Perform optimization step with hooks."""
        # Run pre-optimization hooks
        for hook in self.pre_optimization_hooks:
            hook(self.model, self.optimizer, loss)
            
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Apply gradient clipping hooks
        for hook in self.gradient_clipping_hooks:
            hook(self.model)
            
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step if available
        if self.scheduler:
            self.scheduler.step()
            
        # Run post-optimization hooks
        for hook in self.post_optimization_hooks:
            hook(self.model, self.optimizer, loss)
            
    def register_pre_optimization_hook(self, hook: Callable) -> None:
        """Register a hook to run before optimization step."""
        self.pre_optimization_hooks.append(hook)
        
    def register_post_optimization_hook(self, hook: Callable) -> None:
        """Register a hook to run after optimization step."""
        self.post_optimization_hooks.append(hook)
        
    def register_gradient_clipping_hook(self, hook: Callable) -> None:
        """Register a hook for gradient clipping."""
        self.gradient_clipping_hooks.append(hook)
        
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
        
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer and scheduler state dicts."""
        state = {
            'optimizer': self.optimizer.state_dict()
        }
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()
        return state
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer and scheduler state dicts."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler and 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler']) 