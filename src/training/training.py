from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from ..core.system_config import config_manager
from ..core.system_logging import logger
from ..models.model_loader import model_loader

class TrainingManager:
    """Manages model training and optimization."""
    
    def __init__(self):
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.scaffold_model = model_loader.scaffold_model
    
    def setup_optimizer(self, num_training_steps: int) -> None:
        """
        Set up the optimizer and learning rate scheduler.
        
        Args:
            num_training_steps: Total number of training steps
        """
        try:
            training_config = config_manager.get_value("training_config")
            
            # Set up optimizer
            self.optimizer = AdamW(
                self.scaffold_model.parameters(),
                lr=training_config["learning_rate"],
                weight_decay=0.01
            )
            
            # Set up scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps
            )
            
            logger.log_info("Optimizer and scheduler set up successfully")
            
        except Exception as e:
            logger.log_error(f"Error setting up optimizer: {e}")
            raise
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing input tensors
            
        Returns:
            Tuple of (loss, accuracy)
        """
        try:
            self.scaffold_model.train()
            
            # Forward pass
            outputs = self.scaffold_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == batch["labels"]).float().mean()
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            return loss.item(), accuracy.item()
            
        except Exception as e:
            logger.log_error(f"Error in training step: {e}")
            raise
    
    def run_training_cycle(
        self,
        train_data: List[Dict[str, torch.Tensor]],
        valid_data: List[Dict[str, torch.Tensor]],
        epochs: int = 3,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Run a complete training cycle.
        
        Args:
            train_data: List of training batches
            valid_data: List of validation batches
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            metrics = {
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": []
            }
            
            # Set up optimizer if not already done
            if self.optimizer is None:
                num_training_steps = len(train_data) * epochs
                self.setup_optimizer(num_training_steps)
            
            for epoch in range(epochs):
                # Training phase
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                
                for batch in train_data:
                    loss, accuracy = self.train_step(batch)
                    epoch_loss += loss
                    epoch_accuracy += accuracy
                
                metrics["train_loss"].append(epoch_loss / len(train_data))
                metrics["train_accuracy"].append(epoch_accuracy / len(train_data))
                
                # Validation phase
                val_metrics = self.validate_epoch(valid_data)
                metrics["val_loss"].append(val_metrics["loss"])
                metrics["val_accuracy"].append(val_metrics["accuracy"])
                
                logger.log_info(
                    f"Epoch {epoch + 1}/{epochs}",
                    {
                        "train_loss": metrics["train_loss"][-1],
                        "train_accuracy": metrics["train_accuracy"][-1],
                        "val_loss": metrics["val_loss"][-1],
                        "val_accuracy": metrics["val_accuracy"][-1]
                    }
                )
            
            return metrics
            
        except Exception as e:
            logger.log_error(f"Error in training cycle: {e}")
            raise
    
    def validate_epoch(self, valid_data: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Validate the model on a validation dataset.
        
        Args:
            valid_data: List of validation batches
            
        Returns:
            Dictionary containing validation metrics
        """
        try:
            self.scaffold_model.eval()
            total_loss = 0.0
            total_accuracy = 0.0
            
            with torch.no_grad():
                for batch in valid_data:
                    outputs = self.scaffold_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    preds = torch.argmax(logits, dim=-1)
                    accuracy = (preds == batch["labels"]).float().mean()
                    
                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
            
            return {
                "loss": total_loss / len(valid_data),
                "accuracy": total_accuracy / len(valid_data)
            }
            
        except Exception as e:
            logger.log_error(f"Error in validation: {e}")
            raise

# Create a singleton instance
training_manager = TrainingManager() 