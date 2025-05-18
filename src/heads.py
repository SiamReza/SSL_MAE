"""
Head modules for the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection.
This module provides various head architectures for different downstream tasks.
"""

import torch
import torch.nn as nn
from typing import Optional


class ProjectionHead(nn.Module):
    """
    Projection head for self-supervised learning and downstream tasks.
    
    This head projects the encoder's output into a lower-dimensional space,
    which is useful for linear evaluation and fine-tuning.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        proj_dim: int = 256, 
        hidden_dim: Optional[int] = None,
        use_bn: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize the projection head.
        
        Args:
            input_dim: Input dimension (encoder's output dimension)
            proj_dim: Output projection dimension
            hidden_dim: Hidden dimension (if None, use input_dim)
            use_bn: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        # First layer: input_dim -> hidden_dim
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Second layer: hidden_dim -> proj_dim
        layers.append(nn.Linear(hidden_dim, proj_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(proj_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projection head.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Projected tensor of shape [batch_size, proj_dim]
        """
        return self.net(x)
