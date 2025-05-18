"""
Classifier module for the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection.
This module provides a classifier for morph detection.
"""

import torch
import torch.nn as nn
from typing import Optional


class MorphClassifier(nn.Module):
    """
    Classifier for morph detection.
    
    This classifier takes the encoder's output (or projection head's output)
    and predicts whether an image is bonafide or morph.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int = 2,
        hidden_dim: Optional[int] = None,
        use_bn: bool = True,
        dropout: float = 0.5
    ):
        """
        Initialize the morph classifier.
        
        Args:
            input_dim: Input dimension (encoder's or projection head's output dimension)
            num_classes: Number of output classes (2 for binary classification)
            hidden_dim: Hidden dimension (if None, use input_dim // 2)
            use_bn: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        # Layer normalization
        self.norm = nn.LayerNorm(input_dim)
        
        # First layer: input_dim -> hidden_dim
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Second layer: hidden_dim -> num_classes
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Logits tensor of shape [batch_size, num_classes]
        """
        x = self.norm(x)
        return self.head(x)
