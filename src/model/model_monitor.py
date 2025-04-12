from typing import Dict, Any, Optional, Callable, List
import torch
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from .model_config import ModelConfig

logger = logging.getLogger(__name__)

class ModelMonitor:
    """Handles model monitoring and visualization with extensibility hooks."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        log_dir: str = "logs",
        plot_dir: str = "plots"
    ):
        self.model = model
        self.config = config
        self.log_dir = Path(log_dir)
        self.plot_dir = Path(plot_dir)
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.setup_logging()
        
        # Monitoring state
        self.metrics_history: Dict[str, List[float]] = {}
        self.gradient_history: Dict[str, List[float]] = {}
        self.learning_rate_history: List[float] = []
        
        # Hooks for future extensibility
        self.metric_hooks: List[Callable] = []
        self.gradient_hooks: List[Callable] = []
        self.visualization_hooks: List[Callable] = []
        
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
    def log_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """Log training metrics."""
        # Update metrics history
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(value)
            
        # Log metrics
        logger.info(f"Epoch {epoch} - Metrics: {metrics}")
        
        # Run metric hooks
        for hook in self.metric_hooks:
            hook(metrics, epoch)
            
    def log_gradients(self, epoch: int) -> None:
        """Log model gradients."""
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradients[name] = grad_norm
                
                # Update gradient history
                if name not in self.gradient_history:
                    self.gradient_history[name] = []
                self.gradient_history[name].append(grad_norm)
                
        # Log gradients
        logger.info(f"Epoch {epoch} - Gradient norms: {gradients}")
        
        # Run gradient hooks
        for hook in self.gradient_hooks:
            hook(gradients, epoch)
            
    def log_learning_rate(self, lr: float, epoch: int) -> None:
        """Log learning rate."""
        self.learning_rate_history.append(lr)
        logger.info(f"Epoch {epoch} - Learning rate: {lr}")
        
    def plot_metrics(self, save: bool = True) -> None:
        """Plot training metrics."""
        plt.figure(figsize=(12, 6))
        
        for name, values in self.metrics_history.items():
            plt.plot(values, label=name)
            
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Metrics')
        plt.legend()
        plt.grid(True)
        
        if save:
            plot_path = self.plot_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path)
            plt.close()
            
    def plot_gradients(self, save: bool = True) -> None:
        """Plot gradient norms."""
        plt.figure(figsize=(12, 6))
        
        for name, values in self.gradient_history.items():
            plt.plot(values, label=name)
            
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms')
        plt.legend()
        plt.grid(True)
        
        if save:
            plot_path = self.plot_dir / f"gradients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path)
            plt.close()
            
    def plot_learning_rate(self, save: bool = True) -> None:
        """Plot learning rate history."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.learning_rate_history)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        if save:
            plot_path = self.plot_dir / f"learning_rate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path)
            plt.close()
            
    def save_training_summary(self) -> None:
        """Save training summary to JSON file."""
        summary = {
            'config': self.config.__dict__,
            'metrics_history': self.metrics_history,
            'gradient_history': self.gradient_history,
            'learning_rate_history': self.learning_rate_history
        }
        
        summary_path = self.log_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def register_metric_hook(self, hook: Callable) -> None:
        """Register a hook for metric logging."""
        self.metric_hooks.append(hook)
        
    def register_gradient_hook(self, hook: Callable) -> None:
        """Register a hook for gradient logging."""
        self.gradient_hooks.append(hook)
        
    def register_visualization_hook(self, hook: Callable) -> None:
        """Register a hook for visualization."""
        self.visualization_hooks.append(hook)
        
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Get metrics history."""
        return self.metrics_history
        
    def get_gradient_history(self) -> Dict[str, List[float]]:
        """Get gradient history."""
        return self.gradient_history
        
    def get_learning_rate_history(self) -> List[float]:
        """Get learning rate history."""
        return self.learning_rate_history 