"""
LoRA (Low-Rank Adaptation) module for the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection.
This module provides LoRA implementation for parameter-efficient fine-tuning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple


class LoRALayer(nn.Module):
    """
    LoRA layer implementation.

    This layer adds a low-rank adaptation to a pre-trained weight matrix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0
    ):
        """
        Initialize the LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            r: Rank of the low-rank adaptation
            lora_alpha: Alpha parameter for scaling
            lora_dropout: Dropout probability for LoRA layers
        """
        super().__init__()

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Optional dropout
        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LoRA layer.

        Args:
            x: Input tensor of shape [*, in_features]

        Returns:
            LoRA adaptation of shape [*, out_features]
        """
        # Ensure lora_A and lora_B are on the same device as x
        if self.lora_A.device != x.device:
            self.lora_A = self.lora_A.to(x.device)
            self.lora_B = self.lora_B.to(x.device)

        return (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRA(nn.Module):
    """
    LoRA wrapper for nn.Linear or nn.MultiheadAttention.

    This module wraps a pre-trained layer and adds LoRA adaptation.
    """

    def __init__(
        self,
        layer: Union[nn.Linear, nn.MultiheadAttention],
        r: int = 4,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0
    ):
        """
        Initialize the LoRA wrapper.

        Args:
            layer: Pre-trained layer to adapt
            r: Rank of the low-rank adaptation
            lora_alpha: Alpha parameter for scaling
            lora_dropout: Dropout probability for LoRA layers
        """
        super().__init__()

        self.layer = layer
        self.r = r
        self.lora_alpha = lora_alpha

        # Handle different layer types
        if isinstance(layer, nn.Linear):
            self.lora = LoRALayer(
                layer.in_features,
                layer.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            # Save original forward
            self.original_forward = layer.forward
            # Replace forward method
            layer.forward = self.forward_linear

        elif isinstance(layer, nn.MultiheadAttention):
            # For MultiheadAttention, we need to adapt q, k, v projections
            embed_dim = layer.embed_dim

            # Create LoRA layers for q, k, v projections
            self.q_lora = LoRALayer(
                embed_dim,
                embed_dim,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )

            self.k_lora = LoRALayer(
                embed_dim,
                embed_dim,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )

            self.v_lora = LoRALayer(
                embed_dim,
                embed_dim,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )

            # Save original forward
            self.original_forward = layer.forward
            # Replace forward method
            layer.forward = self.forward_mha

    def forward_linear(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Linear layer with LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            Output tensor with LoRA adaptation
        """
        return self.original_forward(x) + self.lora(x)

    def forward_mha(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for MultiheadAttention with LoRA adaptation.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            key_padding_mask: Mask for keys
            need_weights: Whether to return attention weights
            attn_mask: Attention mask

        Returns:
            Output tensor with LoRA adaptation and attention weights
        """
        # Apply original MHA
        output, attn_weights = self.original_forward(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask
        )

        # Apply LoRA adaptations
        q_output = self.q_lora(query)
        k_output = self.k_lora(key)
        v_output = self.v_lora(value)

        # Get num_heads and head_dim from the layer if available, or calculate them
        if hasattr(self.layer, 'num_heads') and hasattr(self.layer, 'head_dim'):
            num_heads = self.layer.num_heads
            head_dim = self.layer.head_dim
        else:
            # For PyTorch's MultiheadAttention, these might not be directly accessible
            # We can calculate them from embed_dim and num_heads
            embed_dim = self.layer.embed_dim
            num_heads = self.layer.num_heads
            head_dim = embed_dim // num_heads

        # Compute additional attention with LoRA adaptations
        q_output = q_output.reshape(query.shape[0], -1, num_heads, head_dim).transpose(1, 2)
        k_output = k_output.reshape(key.shape[0], -1, num_heads, head_dim).transpose(1, 2)
        v_output = v_output.reshape(value.shape[0], -1, num_heads, head_dim).transpose(1, 2)

        # Compute attention scores
        attn_output_weights = torch.matmul(q_output, k_output.transpose(-2, -1)) / math.sqrt(head_dim)

        if attn_mask is not None:
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output = torch.matmul(attn_output_weights, v_output)

        # Reshape and add to original output
        attn_output = attn_output.transpose(1, 2).reshape(query.shape)
        output = output + attn_output

        return output, attn_weights


def apply_lora(model: nn.Module, r: int = 4, lora_alpha: int = 32, lora_dropout: float = 0.0) -> nn.Module:
    """
    Apply LoRA to a model.

    Args:
        model: Model to apply LoRA to
        r: Rank of the low-rank adaptation
        lora_alpha: Alpha parameter for scaling
        lora_dropout: Dropout probability for LoRA layers

    Returns:
        Model with LoRA applied
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
            LoRA(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    return model
