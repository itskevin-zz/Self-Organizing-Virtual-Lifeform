from typing import Dict, Any, Optional, Callable, List
import torch
import os
import json
import logging
from pathlib import Path
from .model_config import ModelConfig

logger = logging.getLogger(__name__)

class ModelSaver:
    """Handles model saving and checkpointing with extensibility hooks."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        save_dir: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        
        # Create save directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Hooks for future extensibility
        self.pre_save_hooks: List[Callable] = []
        self.post_save_hooks: List[Callable] = []
        self.checkpoint_hooks: List[Callable] = []
        
    def save_model(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_optimizer: bool = True,
        save_scheduler: bool = True
    ) -> Path:
        """Save model and optional components."""
        save_path = self.save_dir / name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Run pre-save hooks
        for hook in self.pre_save_hooks:
            hook(self.model, save_path)
            
        try:
            # Save model state
            model_path = save_path / "model.pt"
            torch.save(self.model.state_dict(), model_path)
            
            # Save optimizer state if available
            if save_optimizer and self.optimizer:
                optimizer_path = save_path / "optimizer.pt"
                torch.save(self.optimizer.state_dict(), optimizer_path)
                
            # Save scheduler state if available
            if save_scheduler and self.scheduler:
                scheduler_path = save_path / "scheduler.pt"
                torch.save(self.scheduler.state_dict(), scheduler_path)
                
            # Save config and metadata
            config_path = save_path / "config.json"
            config_data = {
                "model_config": self.config.__dict__,
                "metadata": metadata or {}
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            # Run post-save hooks
            for hook in self.post_save_hooks:
                hook(self.model, save_path)
                
            logger.info(f"Model saved successfully to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
            
    def save_checkpoint(
        self,
        name: str,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save training checkpoint."""
        checkpoint_path = self.save_dir / "checkpoints" / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Run checkpoint hooks
        for hook in self.checkpoint_hooks:
            hook(self.model, checkpoint_path, epoch, metrics)
            
        try:
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'metrics': metrics,
                'metadata': metadata or {}
            }
            
            if self.optimizer:
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                
            checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_file)
            
            # Save best model if specified
            if is_best:
                best_path = self.save_dir / "best_model.pt"
                torch.save(self.model.state_dict(), best_path)
                
            logger.info(f"Checkpoint saved successfully to {checkpoint_file}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            raise
            
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load training checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise
            
    def register_pre_save_hook(self, hook: Callable) -> None:
        """Register a hook to run before saving."""
        self.pre_save_hooks.append(hook)
        
    def register_post_save_hook(self, hook: Callable) -> None:
        """Register a hook to run after saving."""
        self.post_save_hooks.append(hook)
        
    def register_checkpoint_hook(self, hook: Callable) -> None:
        """Register a hook for checkpoint operations."""
        self.checkpoint_hooks.append(hook)
        
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        checkpoint_dir = self.save_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return None
            
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None
            
        return max(checkpoints, key=lambda x: int(x.stem.split('_')[-1])) 