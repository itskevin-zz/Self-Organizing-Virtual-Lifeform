import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from ..core.system_logging import logger

class SimpleCrossAttentionFuser(nn.Module):
    """Implements cross-attention between base and scaffold models."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Attention layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Control parameters
        self.influence_weight = nn.Parameter(torch.tensor(0.5))
        self.blend_strength = nn.Parameter(torch.tensor(0.5))
    
    def set_influence_weight(self, weight: float) -> None:
        """Set the influence weight of the scaffold model."""
        self.influence_weight.data = torch.tensor(weight)
    
    def set_blend_strength(self, strength: float) -> None:
        """Set the blending strength between base and scaffold outputs."""
        self.blend_strength.data = torch.tensor(strength)
    
    def forward(self, base_hidden_state: torch.Tensor, scaffold_context: torch.Tensor) -> torch.Tensor:
        """
        Perform cross-attention between base and scaffold models.
        
        Args:
            base_hidden_state: Hidden states from the base model
            scaffold_context: Context from the scaffold model
            
        Returns:
            Combined hidden states
        """
        try:
            batch_size = base_hidden_state.size(0)
            
            # Project queries, keys, and values
            q = self.q_proj(base_hidden_state)
            k = self.k_proj(scaffold_context)
            v = self.v_proj(scaffold_context)
            
            # Reshape for multi-head attention
            q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            
            # Apply attention
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, -1, self.hidden_dim)
            
            # Project output
            attn_output = self.out_proj(attn_output)
            
            # Blend with base hidden state
            blended_output = (1 - self.blend_strength) * base_hidden_state + \
                           self.blend_strength * attn_output
            
            # Apply influence weight
            final_output = (1 - self.influence_weight) * base_hidden_state + \
                         self.influence_weight * blended_output
            
            return final_output
            
        except Exception as e:
            logger.log_error(f"Error in cross-attention forward pass: {e}")
            raise

class CrossAttentionManager:
    """Manages cross-attention injection and configuration."""
    
    def __init__(self):
        self.cross_attention_modules: Dict[int, SimpleCrossAttentionFuser] = {}
    
    def insert_cross_attention(
        self,
        model: nn.Module,
        layers: List[int],
        hidden_dim: int,
        num_heads: int = 8
    ) -> None:
        """
        Insert cross-attention modules into specified layers.
        
        Args:
            model: The model to modify
            layers: List of layer indices to modify
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
        """
        try:
            for layer_idx in layers:
                if layer_idx in self.cross_attention_modules:
                    continue
                
                # Create cross-attention module
                cross_attn = SimpleCrossAttentionFuser(hidden_dim, num_heads)
                self.cross_attention_modules[layer_idx] = cross_attn
                
                # Replace the original layer with a modified version
                original_layer = model.transformer.h[layer_idx]
                modified_layer = self._create_modified_layer(original_layer, cross_attn)
                model.transformer.h[layer_idx] = modified_layer
            
            logger.log_info(f"Cross-attention modules inserted at layers: {layers}")
            
        except Exception as e:
            logger.log_error(f"Error inserting cross-attention: {e}")
            raise
    
    def _create_modified_layer(
        self,
        original_layer: nn.Module,
        cross_attn: SimpleCrossAttentionFuser
    ) -> nn.Module:
        """Create a modified layer with cross-attention."""
        class ModifiedLayer(nn.Module):
            def __init__(self, orig_layer, cross_attn_module):
                super().__init__()
                self.original_layer = orig_layer
                self.cross_attn = cross_attn_module
            
            def forward(self, hidden_states, **kwargs):
                # Original layer forward pass
                base_output = self.original_layer(hidden_states, **kwargs)
                
                # Apply cross-attention if scaffold context is available
                if hasattr(self, 'scaffold_context'):
                    base_output = self.cross_attn(base_output, self.scaffold_context)
                
                return base_output
        
        return ModifiedLayer(original_layer, cross_attn)

# Create a singleton instance
cross_attention_manager = CrossAttentionManager() 