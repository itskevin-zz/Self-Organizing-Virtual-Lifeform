from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import time
import math

def setup_optimizer(
    model: PreTrainedModel,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    num_training_steps: int = 1000
) -> Tuple[Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """
    Set up optimizer and learning rate scheduler.
    
    Args:
        model: Model to optimize
        learning_rate: Initial learning rate
        weight_decay: Weight decay coefficient
        warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
        )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler

def train_step(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    optimizer: Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    scaffold_context: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Perform a single training step.
    
    Args:
        model: Model to train
        batch: Input batch
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        scaffold_context: Context from scaffold model (optional)
        
    Returns:
        Dictionary containing training metrics
    """
    model.train()
    
    # Forward pass
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        scaffold_context=scaffold_context
    )
    
    loss = outputs.loss / gradient_accumulation_steps
    loss.backward()
    
    # Gradient clipping
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    # Optimizer step
    if (optimizer.step_count + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
    
    return {
        "loss": loss.item() * gradient_accumulation_steps,
        "learning_rate": optimizer.param_groups[0]["lr"]
    }

def run_training_cycle(
    model: PreTrainedModel,
    train_dataloader: DataLoader,
    valid_dataloader: Optional[DataLoader] = None,
    epochs: int = 3,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    log_interval: int = 10,
    save_interval: int = 100,
    save_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Run a complete training cycle.
    
    Args:
        model: Model to train
        train_dataloader: Training data loader
        valid_dataloader: Validation data loader (optional)
        epochs: Number of epochs
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        log_interval: Interval for logging metrics
        save_interval: Interval for saving checkpoints
        save_path: Path to save checkpoints (optional)
        
    Returns:
        Dictionary containing training history
    """
    if optimizer is None:
        optimizer, scheduler = setup_optimizer(model)
    
    history = {
        "train_loss": [],
        "valid_loss": [],
        "learning_rate": []
    }
    
    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for step, batch in enumerate(train_dataloader):
            # Training step
            metrics = train_step(
                model,
                batch,
                optimizer,
                scheduler,
                gradient_accumulation_steps,
                max_grad_norm
            )
            
            epoch_loss += metrics["loss"]
            global_step += 1
            
            # Logging
            if global_step % log_interval == 0:
                print(f"Step {global_step}: Loss = {metrics['loss']:.4f}, LR = {metrics['learning_rate']:.2e}")
            
            # Save checkpoint
            if save_path and global_step % save_interval == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "global_step": global_step,
                    "epoch": epoch
                }, f"{save_path}/checkpoint_{global_step}.pt")
        
        # Validation
        if valid_dataloader is not None:
            valid_loss = validate_epoch(model, valid_dataloader)
            history["valid_loss"].append(valid_loss)
        
        # Update history
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        history["train_loss"].append(avg_epoch_loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {avg_epoch_loss:.4f}")
        if valid_dataloader is not None:
            print(f"  Valid Loss: {valid_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
    
    return history 