from typing import Optional, Dict, List
import torch
import time
import random
from ..core.system_config import config_manager
from ..core.system_logging import logger
from ..models.model_loader import model_loader
from .training import training_manager

class GestationManager:
    """Manages gestation and sleep training processes."""
    
    def __init__(self):
        self.last_gestation_time = 0
        self.is_gestating = False
        self.scaffold_model = model_loader.scaffold_model
    
    def should_gestate(self) -> bool:
        """
        Check if gestation should occur based on configuration and timing.
        
        Returns:
            bool indicating whether gestation should occur
        """
        try:
            controls_config = config_manager.get_value("controls_config")
            
            # Check if gestation is enabled
            if not controls_config.get("enable_gestation", True):
                return False
            
            # Check time since last gestation
            current_time = time.time()
            time_since_last = current_time - self.last_gestation_time
            min_interval = controls_config.get("gestation_interval", 300)  # 5 minutes default
            
            return time_since_last >= min_interval
            
        except Exception as e:
            logger.log_error(f"Error checking gestation condition: {e}")
            return False
    
    def gestate(self, resume: bool = False) -> Dict[str, float]:
        """
        Perform gestation (training) process.
        
        Args:
            resume: Whether to resume from a previous state
            
        Returns:
            Dictionary containing gestation metrics
        """
        try:
            if self.is_gestating:
                logger.log_info("Gestation already in progress")
                return {}
            
            self.is_gestating = True
            self.last_gestation_time = time.time()
            
            # Get training configuration
            training_config = config_manager.get_value("training_config")
            controls_config = config_manager.get_value("controls_config")
            
            # Prepare training data
            train_data = self._prepare_training_data()
            if not train_data:
                logger.log_info("No data available for gestation")
                return {}
            
            # Split into train and validation sets
            random.shuffle(train_data)
            split_idx = int(len(train_data) * (1 - training_config["valid_split_ratio"]))
            train_set = train_data[:split_idx]
            valid_set = train_data[split_idx:]
            
            # Run training cycle
            metrics = training_manager.run_training_cycle(
                train_set,
                valid_set,
                epochs=training_config["train_epochs"],
                batch_size=training_config["batch_size"]
            )
            
            self.is_gestating = False
            return metrics
            
        except Exception as e:
            self.is_gestating = False
            logger.log_error(f"Error during gestation: {e}")
            raise
    
    def sleep_train(self) -> Dict[str, float]:
        """
        Perform sleep training (background training during idle time).
        
        Returns:
            Dictionary containing sleep training metrics
        """
        try:
            if self.is_gestating:
                logger.log_info("Cannot sleep train while gestating")
                return {}
            
            # Get training configuration
            training_config = config_manager.get_value("training_config")
            controls_config = config_manager.get_value("controls_config")
            
            # Check if sleep training is enabled
            if not controls_config.get("enable_sleep_training", True):
                return {}
            
            # Prepare training data
            train_data = self._prepare_training_data()
            if not train_data:
                return {}
            
            # Run a shorter training cycle for sleep training
            metrics = training_manager.run_training_cycle(
                train_data,
                train_data,  # Use same data for validation during sleep
                epochs=1,  # Single epoch for sleep training
                batch_size=training_config["batch_size"]
            )
            
            return metrics
            
        except Exception as e:
            logger.log_error(f"Error during sleep training: {e}")
            raise
    
    def _prepare_training_data(self) -> List[Dict[str, torch.Tensor]]:
        """
        Prepare training data from logged interactions.
        
        Returns:
            List of training batches
        """
        try:
            # Get logged interactions
            interactions = logger.read()
            if not interactions:
                return []
            
            # Filter for valid training data
            train_data = []
            for interaction in interactions:
                if "prompt" in interaction and "response" in interaction:
                    # Tokenize and prepare batch
                    inputs = model_loader.scaffold_tokenizer(
                        interaction["prompt"],
                        interaction["response"],
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    train_data.append({
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                        "labels": inputs["input_ids"].clone()
                    })
            
            return train_data
            
        except Exception as e:
            logger.log_error(f"Error preparing training data: {e}")
            return []

# Create a singleton instance
gestation_manager = GestationManager() 