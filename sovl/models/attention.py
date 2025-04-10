from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

class SimpleCrossAttentionFuser(nn.Module):
    """
    Simple cross-attention module for fusing base and scaffold model hidden states.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.influence_weight = 1.0
        self.blend_strength = 0.5

    def set_influence_weight(self, weight: float) -> None:
        """
        Set the influence weight for the cross-attention.
        
        Args:
            weight: Influence weight (0.0 to 1.0)
        """
        self.influence_weight = max(0.0, min(1.0, weight))

    def set_blend_strength(self, strength: float) -> None:
        """
        Set the blend strength for the cross-attention.
        
        Args:
            strength: Blend strength (0.0 to 1.0)
        """
        self.blend_strength = max(0.0, min(1.0, strength))

    def forward(
        self,
        base_hidden_state: torch.Tensor,
        scaffold_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-attention between base and scaffold hidden states.
        
        Args:
            base_hidden_state: Hidden state from base model
            scaffold_context: Context from scaffold model
            
        Returns:
            Fused hidden state
        """
        batch_size = base_hidden_state.size(0)
        
        # Project queries, keys, and values
        q = self.q_proj(base_hidden_state).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        k = self.k_proj(scaffold_context).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        v = self.v_proj(scaffold_context).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        attn_output = self.out_proj(attn_output)
        
        # Blend with original hidden state
        blended_output = (
            self.blend_strength * attn_output +
            (1 - self.blend_strength) * base_hidden_state
        )
        
        return blended_output * self.influence_weight

def get_cross_attention_layers(model: PreTrainedModel) -> List[nn.Module]:
    """
    Get all transformer layers that can be modified for cross-attention.
    
    Args:
        model: Model to analyze
        
    Returns:
        List of transformer layers
    """
    layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList):
            layers.extend(module)
    return layers

def insert_cross_attention(
    model: PreTrainedModel,
    cross_attn_module: SimpleCrossAttentionFuser,
    layer_indices: Optional[List[int]] = None
) -> None:
    """
    Insert cross-attention modules into specified layers of a model.
    
    Args:
        model: Model to modify
        cross_attn_module: Cross-attention module to insert
        layer_indices: Indices of layers to modify (None for all layers)
    """
    layers = get_cross_attention_layers(model)
    
    if layer_indices is None:
        layer_indices = list(range(len(layers)))
    
    for idx in layer_indices:
        if 0 <= idx < len(layers):
            orig_layer = layers[idx]
            
            class ModifiedLayer(nn.Module):
                def __init__(self, orig_layer, cross_attn_module):
                    super().__init__()
                    self.orig_layer = orig_layer
                    self.cross_attn = cross_attn_module
                
                def forward(self, hidden_states, **kwargs):
                    # Store original hidden states
                    orig_hidden = hidden_states
                    
                    # Apply original layer
                    hidden_states = self.orig_layer(hidden_states, **kwargs)
                    
                    # Apply cross-attention if scaffold context is provided
                    if "scaffold_context" in kwargs:
                        hidden_states = self.cross_attn(
                            hidden_states,
                            kwargs["scaffold_context"]
                        )
                    
                    return hidden_states
            
            # Replace the original layer with the modified one
            layers[idx] = ModifiedLayer(orig_layer, cross_attn_module) 