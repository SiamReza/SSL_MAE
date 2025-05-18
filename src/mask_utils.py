"""
Masking utilities for the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection.
This module provides functions for masking patches in images for MAE training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def random_masking(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform random masking by per-sample shuffling.
    
    Args:
        x: Input tensor of shape [batch_size, num_patches, embed_dim]
        mask_ratio: Ratio of patches to mask
        
    Returns:
        Tuple of (masked tensor, mask, ids_restore)
        - masked tensor: [batch_size, num_visible_patches, embed_dim]
        - mask: [batch_size, num_patches], 0 is keep, 1 is remove
        - ids_restore: [batch_size, num_patches]
    """
    batch_size, num_patches, embed_dim = x.shape
    
    # Generate random noise for shuffling
    noise = torch.rand(batch_size, num_patches, device=x.device)  # noise in [0, 1]
    
    # Sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # Keep the first 1-mask_ratio patches, remove the remaining
    keep_len = int(num_patches * (1 - mask_ratio))
    
    # Generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([batch_size, num_patches], device=x.device)
    mask[:, :keep_len] = 0
    
    # Unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    # Get the unmasked patches
    ids_keep = ids_shuffle[:, :keep_len]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, embed_dim))
    
    return x_masked, mask, ids_restore


def apply_mask(
    x: torch.Tensor, 
    mask_ratio: float, 
    cls_token: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply random masking to a sequence of patch embeddings.
    
    Args:
        x: Input tensor of shape [batch_size, num_patches, embed_dim]
        mask_ratio: Ratio of patches to mask
        cls_token: Optional CLS token of shape [batch_size, 1, embed_dim]
        
    Returns:
        Tuple of (masked tensor with CLS token, mask, ids_restore)
        - masked tensor: [batch_size, 1 + num_visible_patches, embed_dim]
        - mask: [batch_size, num_patches], 0 is keep, 1 is remove
        - ids_restore: [batch_size, num_patches]
    """
    # Remove CLS token if present
    if cls_token is not None:
        x_patches = x[:, 1:, :]
    else:
        x_patches = x
    
    # Apply random masking
    x_masked, mask, ids_restore = random_masking(x_patches, mask_ratio)
    
    # Add back CLS token if it was provided
    if cls_token is not None:
        x_masked = torch.cat([cls_token, x_masked], dim=1)
    
    return x_masked, mask, ids_restore
