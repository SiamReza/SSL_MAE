"""
MixStyle implementation for domain generalization.
Adapted from https://github.com/KaiyangZhou/mixstyle-release
"""

import random
import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """MixStyle layer for domain generalization.

    Reference:
    Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Initialize MixStyle.

        Args:
            p: Probability of applying mixstyle
            alpha: Parameter for beta distribution
            eps: Small constant for numerical stability
            mix: Mixing strategy ('random', 'crossdomain', or 'within')
        """
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activated(self, activated=True):
        """
        Set whether MixStyle is activated.

        Args:
            activated: Whether to activate MixStyle
        """
        self._activated = activated

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W] or [B, L, C]

        Returns:
            Mixed tensor of the same shape
        """
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        # Get tensor shape
        if x.dim() == 4:  # [B, C, H, W]
            B, C, H, W = x.size()
            x_reshape = x.reshape(B, C, -1)
        else:  # [B, L, C]
            B, seq_len, C = x.size()
            x_reshape = x.permute(0, 2, 1)  # [B, C, seq_len]

        # Calculate mean and variance
        mu = x_reshape.mean(dim=2, keepdim=True)
        var = x_reshape.var(dim=2, keepdim=True)
        sig = (var + self.eps).sqrt()

        # Normalize
        x_normed = (x_reshape - mu) / sig

        # Sample mixing coefficient from beta distribution
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x.device)

        # Get random indices for style mixing
        if self.mix == 'random':
            # Random shuffle
            perm = torch.randperm(B)
        elif self.mix == 'crossdomain':
            # Shuffle within the same domain
            perm = torch.randperm(B)
        else:
            # Shuffle within the same domain
            perm = torch.arange(B)

        # Get mean and variance of other samples
        mu2, sig2 = mu[perm], sig[perm]

        # Mix styles
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        # Denormalize with mixed statistics
        x_mixed = x_normed * sig_mix + mu_mix

        # Reshape back
        if x.dim() == 4:
            x_mixed = x_mixed.reshape(B, C, H, W)
        else:
            x_mixed = x_mixed.permute(0, 2, 1)  # [B, L, C]

        return x_mixed


class ModelWithMixStyle(nn.Module):
    """
    Wrapper to apply MixStyle to a model.

    This wrapper applies MixStyle after specific layers in the model.
    """

    def __init__(self, base_model, p=0.5, alpha=0.1, layers=None):
        """
        Initialize ModelWithMixStyle.

        Args:
            base_model: Base model to wrap
            p: Probability of applying mixstyle
            alpha: Parameter for beta distribution
            layers: List of layer indices to apply MixStyle after (None = all layers)
        """
        super().__init__()
        self.base_model = base_model
        self.p = p
        self.alpha = alpha
        self.layers = layers
        self.mixstyles = nn.ModuleList()

        # Create MixStyle modules
        if layers is None:
            # Apply to all transformer blocks
            num_layers = getattr(base_model, 'num_layers', 12)
            self.layers = list(range(num_layers))

        for _ in range(len(self.layers)):
            self.mixstyles.append(MixStyle(p=p, alpha=alpha))

    def forward(self, x, **kwargs):
        """
        Forward pass with MixStyle applied at specified layers.

        Args:
            x: Input tensor
            **kwargs: Additional arguments to pass to the base model

        Returns:
            Output of the base model with MixStyle applied
        """
        # For ViT-based models, we apply MixStyle after the patch embedding
        # and after specific transformer blocks

        # This is a simplified implementation that assumes the base model
        # exposes intermediate activations. You may need to modify this
        # based on your specific model architecture.

        if hasattr(self.base_model, 'forward_with_mixstyle'):
            # If the base model has a custom forward method for MixStyle
            return self.base_model.forward_with_mixstyle(x, self.mixstyles, self.layers, **kwargs)
        else:
            # Default implementation - just apply MixStyle to the input
            # and pass it through the base model
            x = self.mixstyles[0](x)
            return self.base_model(x, **kwargs)
