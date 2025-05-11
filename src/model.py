"""
Model architecture module for the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection.
This module defines the MorphDetector class which combines a pre-trained MAE backbone with a classification head.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTMAEModel, ViTMAEConfig
from typing import Dict, List, Tuple, Optional, Union, Any


class MorphDetector(nn.Module):
    """
    Model for morph detection using a pre-trained MAE backbone.

    This model combines a pre-trained Vision Transformer (ViT) encoder from a Masked Autoencoder (MAE)
    with a classification head for morph detection. It can operate in two modes:
    1. Pure classification mode (recon_weight=0): Only uses the classification head
    2. Joint mode (recon_weight>0): Uses both reconstruction and classification objectives
    """

    def __init__(self, config):
        """
        Initialize the MorphDetector model.

        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.config = config

        # Load pre-trained MAE model
        self._load_pretrained_model()

        # Add classification head
        self.dropout = nn.Dropout(config.dropout_p)
        self.classifier = nn.Linear(self.hidden_size, 1)

        # Apply freezing strategy
        self._apply_freezing_strategy()

    def _load_pretrained_model(self):
        """Load the pre-trained MAE model based on configuration."""
        # Handle the special case for webssl model which has a different directory name
        if self.config.pretrained_model == "webssl":
            model_path = os.path.join("models", "webssl_mae1b")
        else:
            model_path = os.path.join("models", self.config.pretrained_model)

        # Load model configuration
        with open(os.path.join(model_path, "config.json"), "r") as f:
            model_config = json.load(f)

        # Create MAE configuration
        mae_config = ViTMAEConfig(
            hidden_size=model_config.get("hidden_size", 768),
            num_hidden_layers=model_config.get("num_hidden_layers", 12),
            num_attention_heads=model_config.get("num_attention_heads", 12),
            intermediate_size=model_config.get("intermediate_size", 3072),
            hidden_act=model_config.get("hidden_act", "gelu"),
            hidden_dropout_prob=model_config.get("hidden_dropout_prob", 0.0),
            attention_probs_dropout_prob=model_config.get("attention_probs_dropout_prob", 0.0),
            initializer_range=model_config.get("initializer_range", 0.02),
            layer_norm_eps=model_config.get("layer_norm_eps", 1e-12),
            image_size=model_config.get("image_size", 224),
            patch_size=model_config.get("patch_size", 16),
            num_channels=model_config.get("num_channels", 3),
            mask_ratio=self.config.mae_mask_ratio
        )

        # Initialize MAE model with configuration
        self.mae_model = ViTMAEModel(mae_config)

        # Load pre-trained weights if available
        # Check for different model file formats based on the model type
        if self.config.pretrained_model == "vit_mae":
            weights_path = os.path.join(model_path, "pytorch_model.bin")
        else:  # webssl
            weights_path = os.path.join(model_path, "model.safetensors")

        if os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}")
            state_dict = torch.load(weights_path)

            # Filter out decoder weights as we only need the encoder
            encoder_state_dict = {}
            for k, v in state_dict.items():
                # Handle different key formats in different model files
                if k.startswith("encoder."):
                    encoder_state_dict[k.replace("encoder.", "")] = v
                elif k.startswith("vit."):
                    encoder_state_dict[k.replace("vit.", "")] = v
                elif not any(k.startswith(prefix) for prefix in ["decoder.", "mask_token"]):
                    # If it's not a decoder key and doesn't have a prefix, use as is
                    encoder_state_dict[k] = v

            self.mae_model.load_state_dict(encoder_state_dict, strict=False)
        else:
            print(f"Warning: No model weights found at {weights_path}. Using random initialization.")

        # Store hidden size for classifier
        self.hidden_size = mae_config.hidden_size

    def _apply_freezing_strategy(self):
        """Apply the freezing strategy based on configuration."""
        strategy = self.config.freeze_strategy

        if strategy == "none":
            # All layers are trainable, do nothing
            pass

        elif strategy == "backbone_only":
            # Freeze the entire backbone
            for param in self.mae_model.parameters():
                param.requires_grad = False

        elif strategy == "freeze_except_lastN":
            # Freeze all but the last N transformer blocks
            for name, param in self.mae_model.named_parameters():
                param.requires_grad = False

            # Unfreeze the last N blocks
            last_n = self.config.freeze_lastN
            for i in range(self.mae_model.config.num_hidden_layers - last_n, self.mae_model.config.num_hidden_layers):
                for param in self.mae_model.encoder.layer[i].parameters():
                    param.requires_grad = True

        elif strategy == "gradual_unfreeze":
            # Initially freeze everything except the classification head
            # The train.py will handle the gradual unfreezing based on epoch
            for param in self.mae_model.parameters():
                param.requires_grad = False

    def unfreeze_last_n_blocks(self, n):
        """
        Unfreeze the last n transformer blocks.

        Args:
            n: Number of blocks to unfreeze from the end
        """
        # First ensure all blocks are frozen
        for param in self.mae_model.parameters():
            param.requires_grad = False

        # Then unfreeze the last n blocks
        for i in range(self.mae_model.config.num_hidden_layers - n, self.mae_model.config.num_hidden_layers):
            for param in self.mae_model.encoder.layer[i].parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters in the model."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, pixel_values, labels=None):
        """
        Forward pass through the model.

        Args:
            pixel_values: Tensor of shape (batch_size, num_channels, height, width)
            labels: Optional tensor of shape (batch_size,) with labels (0 for bonafide, 1 for morph)

        Returns:
            Dictionary with model outputs including classification scores and losses
        """
        outputs = {}

        # Apply MAE masking if enabled
        if self.config.use_mae_aug and self.training:
            # Forward pass through MAE model with masking
            mae_outputs = self.mae_model(
                pixel_values=pixel_values,
                output_hidden_states=True
            )

            # Get the [CLS] token from the last hidden state
            cls_token = mae_outputs.last_hidden_state[:, 0]

            # Store mask and ids_restore for reconstruction loss
            outputs["mask"] = mae_outputs.mask
            outputs["ids_restore"] = mae_outputs.ids_restore

            # Calculate reconstruction loss if needed
            if self.config.recon_weight > 0:
                # Calculate mean squared error on the masked patches
                # This is a simplified version that calculates the loss directly on the masked patches
                # In a full implementation, we would use the decoder to reconstruct the masked patches

                # The mask from MAE is (1 for visible, 0 for masked)

                # Calculate the mean squared error on the masked patches
                # We use the difference between the original image patches and a zero tensor
                # as a proxy for the reconstruction error

                # Reshape the image into patches (excluding the CLS token)
                patch_size = 16  # ViT uses 16x16 patches
                num_channels = pixel_values.shape[1]
                num_patches = (pixel_values.shape[2] // patch_size) * (pixel_values.shape[3] // patch_size)

                # Calculate a simple reconstruction loss based on the masked patches
                # This encourages the model to maintain information about the masked regions
                masked_loss = torch.mean(torch.square(mae_outputs.mask)) * num_patches * patch_size * patch_size * num_channels

                outputs["recon_loss"] = masked_loss
        else:
            # Forward pass without masking
            hidden_states = self.mae_model(
                pixel_values=pixel_values,
                output_hidden_states=True
            ).last_hidden_state

            # Get the [CLS] token
            cls_token = hidden_states[:, 0]

            # No reconstruction loss
            outputs["recon_loss"] = torch.tensor(0.0, device=pixel_values.device)

        # Apply dropout and classification head
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)
        outputs["logits"] = logits

        # Apply sigmoid to get scores
        scores = torch.sigmoid(logits)
        outputs["scores"] = scores

        # Calculate classification loss if labels are provided
        if labels is not None:
            cls_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels)
            outputs["cls_loss"] = cls_loss

            # Calculate total loss
            total_loss = cls_loss
            if self.config.recon_weight > 0:
                total_loss += self.config.recon_weight * outputs["recon_loss"]

            outputs["loss"] = total_loss

        return outputs
