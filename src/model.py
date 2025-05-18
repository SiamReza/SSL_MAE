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

from heads import ProjectionHead
from classifier import MorphClassifier
from lora import apply_lora

# We'll use our own implementation of MarginLoss instead of torchmetrics
# This avoids compatibility issues with different versions of torchmetrics
MARGIN_LOSS_AVAILABLE = False

# Custom implementation of MarginLoss
class MarginLoss(nn.Module):
    def __init__(self, margin=0.3, weight=None):
        super().__init__()
        self.margin = margin
        self.weight = weight

    def forward(self, logits, target):
        # Simple margin loss implementation
        # For binary classification: max(0, margin - y * f(x))
        if logits.shape[-1] == 2:  # Two-class logits
            pos_logits = logits[:, 1]
            neg_logits = logits[:, 0]
        else:  # Single output
            pos_logits = logits.squeeze()
            neg_logits = -logits.squeeze()

        # Convert target to -1/1
        target_pm = 2 * target.float() - 1

        # Calculate margin term
        margin_term = self.margin - target_pm * (pos_logits - neg_logits)

        # Clamp to ensure non-negative values
        losses = torch.clamp(margin_term, min=0)

        # Calculate loss
        if self.weight is not None:
            # Apply class weights - make sure weight tensor is on the same device as target
            if self.weight.device != target.device:
                self.weight = self.weight.to(target.device)

            weight_per_sample = self.weight[target.long()]
            loss = (losses * weight_per_sample).mean()
        else:
            # No weights
            loss = losses.mean()

        return loss


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

        # Add learned mask token for decoder if enabled
        if hasattr(config, 'use_learned_mask_token') and config.use_learned_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
            nn.init.normal_(self.mask_token, std=0.02)

        # Add separate positional embeddings for decoder if enabled
        if hasattr(config, 'use_separate_pos_embed') and config.use_separate_pos_embed:
            # Calculate number of patches
            image_size = self.mae_model.config.image_size
            patch_size = self.mae_model.config.patch_size
            num_patches = (image_size // patch_size) ** 2

            # Create separate positional embeddings for decoder
            self.pos_embed_dec = nn.Parameter(torch.zeros(1, num_patches + 1, self.hidden_size))
            nn.init.normal_(self.pos_embed_dec, std=0.02)

        # Add projection head if enabled
        if hasattr(config, 'use_projection_head') and config.use_projection_head:
            proj_dim = getattr(config, 'proj_dim', 256)
            proj_hidden_dim = getattr(config, 'proj_hidden_dim', self.hidden_size // 2)
            self.projection_head = ProjectionHead(
                input_dim=self.hidden_size,
                proj_dim=proj_dim,
                hidden_dim=proj_hidden_dim,
                use_bn=True,
                dropout=config.dropout_p / 2
            )

        # Add dropout
        self.dropout = nn.Dropout(config.dropout_p)

        # Add classification head
        if hasattr(config, 'use_improved_classifier') and config.use_improved_classifier:
            cls_input_dim = getattr(config, 'proj_dim', 256) if config.use_projection_head else self.hidden_size
            cls_hidden_dim = getattr(config, 'cls_hidden_dim', None)
            self.classifier = MorphClassifier(
                input_dim=cls_input_dim,
                num_classes=1,  # Binary classification
                hidden_dim=cls_hidden_dim,
                use_bn=True,
                dropout=config.dropout_p
            )
        else:
            self.classifier = nn.Linear(self.hidden_size, 1)

        # Initialize margin loss if enabled
        if hasattr(config, 'use_margin_loss') and config.use_margin_loss:
            class_weights = getattr(config, 'class_weights', [1.0, 1.0])
            margin_value = getattr(config, 'margin', 0.3)

            # Create margin loss with our custom implementation
            # Store weights as a tensor on CPU initially, will be moved to correct device during forward pass
            class_weights_tensor = torch.tensor(class_weights)
            self.margin_loss = MarginLoss(
                margin=margin_value,
                weight=class_weights_tensor
            )

            # Explicitly move the margin loss to the same device as the model
            # This will be done when the model is moved to a device

        # Apply LoRA if enabled
        if hasattr(config, 'use_lora') and config.use_lora:
            lora_r = getattr(config, 'lora_r', 4)
            lora_alpha = getattr(config, 'lora_alpha', 32)
            lora_dropout = getattr(config, 'lora_dropout', 0.1)
            apply_lora(
                self.mae_model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )

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

    def _reconstruct_from_masked(self, x, mask, ids_restore):
        """
        Reconstruct the masked patches using the decoder.

        Args:
            x: Tensor of visible patches [B, N_vis, D]
            mask: Binary mask [B, N], 0 is keep, 1 is remove
            ids_restore: Indices for restoring the original order [B, N]

        Returns:
            Reconstructed patches [B, N, D]
        """
        batch_size, _, dim = x.shape

        # Calculate number of patches
        image_size = self.mae_model.config.image_size
        patch_size = self.mae_model.config.patch_size
        num_patches = (image_size // patch_size) ** 2

        # Create a tensor to hold all tokens (visible and masked)
        tokens = torch.zeros(batch_size, num_patches + 1, dim, device=x.device)

        # Add CLS token
        cls_token = x[:, 0:1, :]
        tokens[:, 0:1, :] = cls_token

        # Add visible tokens
        visible_tokens = x[:, 1:, :]

        # Create a mask for visible tokens (complement of the input mask, excluding CLS token)
        visible_mask = 1 - mask

        # Use ids_restore to place visible tokens in their original positions

        # Place visible tokens in their original positions
        # First, handle the CLS token separately (it's already placed)

        # Then place the visible tokens (excluding CLS) at their original positions
        # We need to map from the condensed visible tokens to their original positions
        for b in range(batch_size):
            visible_idx = 0  # Reset for each batch
            for i in range(num_patches):
                if visible_mask[b, i] == 1:  # This is a visible token
                    if visible_idx < visible_tokens.shape[1]:  # Check bounds
                        tokens[b, ids_restore[b, i]] = visible_tokens[b, visible_idx]
                        visible_idx += 1
                else:  # This is a masked token
                    tokens[b, ids_restore[b, i]] = self.mask_token  # Use mask token

        # Apply positional embeddings if using separate ones for decoder
        if hasattr(self, 'pos_embed_dec'):
            tokens = tokens + self.pos_embed_dec

        # Simple decoder: just a few transformer layers
        # For simplicity, we'll use the same transformer blocks as the encoder
        # In a full implementation, you would have a separate decoder
        decoder_depth = getattr(self.config, 'decoder_depth', 4)
        for i in range(min(decoder_depth, self.mae_model.config.num_hidden_layers)):
            tokens = self.mae_model.encoder.layer[i](tokens)[0]

        # Final layer norm
        tokens = self.mae_model.layernorm(tokens)

        return tokens

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
            cls_token = mae_outputs.last_hidden_state[:, 0:1]  # Keep dimension for concatenation

            # Store mask and ids_restore for reconstruction loss
            outputs["mask"] = mae_outputs.mask
            outputs["ids_restore"] = mae_outputs.ids_restore

            # Calculate reconstruction loss if needed
            if self.config.recon_weight > 0:
                # Check if we should use the learned mask token and decoder
                if hasattr(self, 'mask_token') and hasattr(self.config, 'use_learned_mask_token') and self.config.use_learned_mask_token:
                    # Reconstruct the masked patches
                    reconstructed = self._reconstruct_from_masked(
                        mae_outputs.last_hidden_state,
                        mae_outputs.mask,
                        mae_outputs.ids_restore
                    )

                    # Calculate reconstruction loss (MSE between reconstructed and original patches)
                    # For simplicity, we'll just use the masked patches
                    masked_indices = torch.nonzero(mae_outputs.mask)

                    # Extract the reconstructed patches at masked positions
                    batch_indices = masked_indices[:, 0]
                    patch_indices = masked_indices[:, 1]

                    # Check if there are any masked patches
                    if len(batch_indices) > 0:
                        reconstructed_patches = reconstructed[batch_indices, patch_indices]

                        # Calculate MSE loss
                        # In a real implementation, you would compare with the original image patches
                        # Here we'll just use a simplified loss
                        masked_loss = torch.mean(torch.square(reconstructed_patches))
                    else:
                        # No masked patches, set loss to zero
                        masked_loss = torch.tensor(0.0, device=pixel_values.device)
                else:
                    # Use the simplified reconstruction loss from the original implementation
                    patch_size = 16  # ViT uses 16x16 patches
                    num_channels = pixel_values.shape[1]
                    num_patches = (pixel_values.shape[2] // patch_size) * (pixel_values.shape[3] // patch_size)

                    # Calculate a simple reconstruction loss based on the masked patches
                    masked_loss = torch.mean(torch.square(mae_outputs.mask)) * num_patches * patch_size * patch_size * num_channels

                outputs["recon_loss"] = masked_loss
        else:
            # Forward pass without masking
            hidden_states = self.mae_model(
                pixel_values=pixel_values,
                output_hidden_states=True
            ).last_hidden_state

            # Get the [CLS] token
            cls_token = hidden_states[:, 0:1]  # Keep dimension for concatenation

            # No reconstruction loss
            outputs["recon_loss"] = torch.tensor(0.0, device=pixel_values.device)

        # Get the final representation
        representation = cls_token.squeeze(1)  # Remove the extra dimension

        # Apply projection head if enabled
        if hasattr(self, 'projection_head') and hasattr(self.config, 'use_projection_head') and self.config.use_projection_head:
            representation = self.projection_head(representation)

        # Apply dropout
        representation = self.dropout(representation)

        # Apply classification head
        logits = self.classifier(representation)
        outputs["logits"] = logits

        # Apply sigmoid to get scores
        scores = torch.sigmoid(logits)
        outputs["scores"] = scores

        # Calculate classification loss if labels are provided
        if labels is not None:
            # Use margin loss if enabled
            if hasattr(self, 'margin_loss') and hasattr(self.config, 'use_margin_loss') and self.config.use_margin_loss:
                # Convert binary labels to class indices (0 or 1)
                label_indices = labels.long()

                # Note: MarginLoss already has a weight parameter set during initialization

                # Use our custom implementation instead of torchmetrics
                # The torchmetrics implementation has different expectations
                if logits.shape[-1] == 1:  # Binary classification with single output
                    # Create two-class logits: [batch_size, 2]
                    two_class_logits = torch.cat([-logits, logits], dim=-1)
                    cls_loss = self.margin_loss(two_class_logits, label_indices)
                else:
                    cls_loss = self.margin_loss(logits, label_indices)
            else:
                # Use standard binary cross-entropy loss with class weights
                if hasattr(self.config, 'class_weights') and len(self.config.class_weights) == 2:
                    # Apply class weights to BCE loss
                    pos_weight = torch.tensor([self.config.class_weights[1] / self.config.class_weights[0]],
                                             device=logits.device)
                    cls_loss = F.binary_cross_entropy_with_logits(
                        logits.squeeze(), labels, pos_weight=pos_weight
                    )
                else:
                    # Standard BCE loss without weights
                    cls_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels)

            outputs["cls_loss"] = cls_loss

            # Calculate total loss
            total_loss = cls_loss
            if self.config.recon_weight > 0:
                total_loss += self.config.recon_weight * outputs["recon_loss"]

            outputs["loss"] = total_loss

        return outputs
