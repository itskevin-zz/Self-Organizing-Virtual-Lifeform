from typing import Dict, List, Optional, Set
import torch
import time
import psutil
from collections import defaultdict
from ..core.system_config import config_manager
from ..core.system_logging import logger

class MemoryManager:
    """Manages system memory and model memory states."""
    
    def __init__(self):
        self.token_map: Dict[int, int] = {}
        self.token_map_memory: Dict[int, float] = {}
        self.scaffold_memory: Dict[str, float] = {}
        self.memory_timestamps: Dict[str, float] = {}
        self.active_tokens: Set[int] = set()
    
    def check_memory_health(self) -> bool:
        """
        Check system memory health and perform cleanup if necessary.
        
        Returns:
            bool indicating whether memory is healthy
        """
        try:
            controls_config = config_manager.get_value("controls_config")
            memory_threshold = controls_config.get("memory_threshold", 0.85)
            
            # Check system memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            if memory_usage > memory_threshold:
                logger.log_info(f"High memory usage detected: {memory_usage:.2%}")
                self._cleanup_memory()
                
                # Check if cleanup was successful
                memory = psutil.virtual_memory()
                memory_usage = memory.percent / 100.0
                
                if memory_usage > memory_threshold:
                    logger.log_error("Memory usage still high after cleanup")
                    return False
            
            return True
            
        except Exception as e:
            logger.log_error(f"Error checking memory health: {e}")
            return False
    
    def update_token_map_memory(self, token_id: int, confidence: float) -> None:
        """
        Update token mapping memory with confidence scores.
        
        Args:
            token_id: Token ID to update
            confidence: Confidence score for the token
        """
        try:
            if token_id not in self.token_map_memory:
                self.token_map_memory[token_id] = 0.0
            
            # Update with confidence-weighted value
            self.token_map_memory[token_id] = max(
                self.token_map_memory[token_id],
                confidence
            )
            
            # Add to active tokens
            self.active_tokens.add(token_id)
            
        except Exception as e:
            logger.log_error(f"Error updating token map memory: {e}")
    
    def update_scaffold_memory(self, key: str, value: float) -> None:
        """
        Update scaffold model memory with new values.
        
        Args:
            key: Memory key
            value: Memory value
        """
        try:
            self.scaffold_memory[key] = value
            self.memory_timestamps[key] = time.time()
            
        except Exception as e:
            logger.log_error(f"Error updating scaffold memory: {e}")
    
    def decay_memories(self) -> None:
        """Apply decay to all memory types."""
        try:
            controls_config = config_manager.get_value("controls_config")
            decay_rate = controls_config.get("memory_decay_rate", 0.95)
            
            # Decay token map memory
            for token_id in list(self.token_map_memory.keys()):
                self.token_map_memory[token_id] *= decay_rate
                
                # Remove if below threshold
                if self.token_map_memory[token_id] < 0.1:
                    del self.token_map_memory[token_id]
                    if token_id in self.token_map:
                        del self.token_map[token_id]
            
            # Decay scaffold memory
            current_time = time.time()
            for key in list(self.scaffold_memory.keys()):
                age = current_time - self.memory_timestamps[key]
                decay = decay_rate ** (age / 3600)  # Decay based on hours
                self.scaffold_memory[key] *= decay
                
                # Remove if below threshold
                if self.scaffold_memory[key] < 0.1:
                    del self.scaffold_memory[key]
                    del self.memory_timestamps[key]
            
        except Exception as e:
            logger.log_error(f"Error decaying memories: {e}")
    
    def _cleanup_memory(self) -> None:
        """Perform memory cleanup operations."""
        try:
            # Clear inactive tokens
            current_time = time.time()
            inactive_tokens = set(self.token_map_memory.keys()) - self.active_tokens
            for token_id in inactive_tokens:
                del self.token_map_memory[token_id]
                if token_id in self.token_map:
                    del self.token_map[token_id]
            
            # Reset active tokens
            self.active_tokens.clear()
            
            # Apply aggressive decay
            controls_config = config_manager.get_value("controls_config")
            aggressive_decay = controls_config.get("memory_aggressive_decay", 0.5)
            
            # Decay all memories
            for key in self.scaffold_memory:
                self.scaffold_memory[key] *= aggressive_decay
            
            for token_id in list(self.token_map_memory.keys()):
                self.token_map_memory[token_id] *= aggressive_decay
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.log_info("Memory cleanup completed")
            
        except Exception as e:
            logger.log_error(f"Error during memory cleanup: {e}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        try:
            stats = {
                "system_memory_usage": psutil.virtual_memory().percent / 100.0,
                "token_map_size": len(self.token_map),
                "token_memory_size": len(self.token_map_memory),
                "scaffold_memory_size": len(self.scaffold_memory),
                "active_tokens": len(self.active_tokens)
            }
            
            if torch.cuda.is_available():
                stats["gpu_memory_usage"] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            return stats
            
        except Exception as e:
            logger.log_error(f"Error getting memory stats: {e}")
            return {}

# Create a singleton instance
memory_manager = MemoryManager() 