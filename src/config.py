"""
Configuration module for the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection.
This module contains all parameters needed for the project, organized into logical sections.
"""

import os
import argparse
from typing import List, Dict, Any, Optional, Union


class Config:
    """Configuration class for the morph detection project."""

    def __init__(self):
        """Initialize default configuration values."""
        # Dataset Options
        self.data_root = "datasets"
        self.train_dataset = "LMA"  # Default training dataset
        self.test_datasets = ["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN"] # List of datasets to evaluate on
        self.train_val_pct = 0.80  # 85% for training, 15% for validation

        # Training Hyperparameters
        self.num_epochs = 30  # Increased for better convergence on small datasets
        self.batch_size = 32  # Reduced to have more update steps on small datasets
        self.learning_rate = 1e-5  # Slightly increased for faster convergence
        self.weight_decay = 0.1  # Weight decay for regularization
        self.recon_weight = 0.0  # Weight for reconstruction loss (1 for MAE+classifier, 0 for supervised-only)
        self.lr_scheduler = "cosine"  # 'cosine' or 'none'
        self.eta_min = 0.0  # Minimum learning rate at the end of cosine schedule
        self.T_max = self.num_epochs  # Number of epochs over which to decay
        self.use_fixed_seed = True  # Whether to use a fixed seed (True) or random seed (False)
        self.seed = 42  # Seed value when use_fixed_seed is True

        # Early stopping parameters
        self.use_early_stopping = True  # Whether to use early stopping
        self.patience = 5  # Number of epochs to wait for improvement before stopping
        self.min_delta = 0.001  # Minimum change in validation loss to be considered an improvement

        # Regularization and Backbone Control
        self.freeze_strategy = "freeze_except_lastN"  # 'freeze_except_lastN', 'none', 'backbone_only', or 'gradual_unfreeze'
        self.warmup_epochs = 2  # Number of epochs to train only the head (for 'gradual_unfreeze')
        self.mid_epochs = 5  # Epoch to start unfreezing the full backbone (for 'gradual_unfreeze')
        self.freeze_lastN = 12  # Number of final ViT blocks to keep trainable

        self.dropout_p = 0.9  # Dropout probability (increased for better regularization)
        self.use_mae_aug = True  # Whether to use MAE-specific augmentations
        self.mae_mask_ratio = 0.75  # Ratio of patches to mask in MAE (0.0 to 1.0)
        self.use_cls_aug = True  # Whether to use classifier augmentations

        # Advanced features
        # Augmentation options
        self.use_advanced_aug = True  # Whether to use advanced augmentations (requires albumentations)
        self.advanced_aug_magnitude = 9  # Magnitude for RandAugment (1-10)
        self.advanced_aug_num_ops = 2  # Number of operations for RandAugment

        # Data loading options
        self.use_balanced_sampler = True  # Whether to use balanced sampling for training
        self.use_subject_splitting = True  # Whether to use subject-based splitting to prevent data leakage

        # Model architecture options
        self.use_projection_head = False  # Whether to use projection head for downstream tasks
        self.proj_dim = 128  # Reduced projection dimension to prevent overfitting
        self.proj_hidden_dim = None  # Hidden dimension for projection head (None = input_dim // 2)
        self.use_improved_classifier = False  # Enabled improved classifier for better performance
        self.cls_hidden_dim = None  # Hidden dimension for classifier (None = input_dim // 2)

        # Loss function options
        self.use_margin_loss = False  # Whether to use margin-based loss
        self.margin = 0.3  # Margin for margin-based loss
        self.class_weights = [1.37, 0.79]  # Class weights for loss function [bonafide_weight, morph_weight]
        # Note: MorDiff dataset has different class distribution with weights [2.73, 0.61]

        # MAE-specific options
        self.use_learned_mask_token = True  # Enabled learned mask token in decoder for better reconstruction
        self.use_separate_pos_embed = True  # Enabled separate positional embeddings for encoder and decoder
        self.decoder_depth = 4  # Number of transformer layers in the decoder

        # LoRA options
        self.use_lora = False  # Whether to use LoRA for parameter-efficient fine-tuning
        self.lora_r = 4  # Rank for LoRA adaptation
        self.lora_alpha = 32  # Alpha parameter for LoRA
        self.lora_dropout = 0.1  # Dropout probability for LoRA layers
        self.lora_target_modules = ["query", "key", "value"]  # Target modules for LoRA adaptation

        # Cross-validation options
        self.use_cross_validation = True  # Whether to use cross-validation
        self.n_folds = 5  # Number of folds for cross-validation

        # Model Selection
        self.pretrained_model = "vit_mae"  # 'vit_mae' or 'webssl'
        # Note: The webssl model is stored in the 'webssl_mae1b' directory, but we use 'webssl' as the identifier

        # Output Configuration
        self.output_dir = "output"
        self.models_dir = os.path.join(self.output_dir, "models")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.final_model_name = "{}_morphdetector.pt"  # Template for saved model name
        self.log_metrics = ["epoch", "train_loss", "val_loss", "val_acc", "learning_rate"]

        # Create output directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def update_from_args(self, args: argparse.Namespace) -> None:
        """
        Update configuration from command-line arguments.

        Args:
            args: Command-line arguments parsed by argparse
        """
        # Process override arguments if they exist
        if hasattr(args, 'override') and args.override:
            for override in args.override:
                if '=' in override:
                    key, value = override.split('=', 1)
                    if hasattr(self, key):
                        # Convert value to the appropriate type
                        orig_value = getattr(self, key)
                        if isinstance(orig_value, bool):
                            setattr(self, key, value.lower() == 'true')
                        elif isinstance(orig_value, int):
                            setattr(self, key, int(value))
                        elif isinstance(orig_value, float):
                            setattr(self, key, float(value))
                        elif isinstance(orig_value, list):
                            # Assume comma-separated list
                            setattr(self, key, value.split(','))
                        else:
                            setattr(self, key, value)
                    else:
                        print(f"Warning: Unknown configuration parameter '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __str__(self) -> str:
        """
        Get string representation of the configuration.

        Returns:
            Formatted string representation
        """
        config_str = "Configuration:\n"
        for key, value in self.to_dict().items():
            config_str += f"  {key}: {value}\n"
        return config_str


def get_config() -> Config:
    """
    Get configuration with command-line overrides.

    Returns:
        Configuration object
    """
    parser = argparse.ArgumentParser(description='Morph Detection Pipeline')
    parser.add_argument('--override', action='append',
                        help='Override configuration parameters, format: key=value')
    parser.add_argument('--mask_ratio', type=float, default=None,
                        help='Ratio of patches to mask in MAE (0.0 to 1.0)')

    args, _ = parser.parse_known_args()

    config = Config()

    # Apply mask_ratio if provided
    if args.mask_ratio is not None:
        config.mae_mask_ratio = args.mask_ratio

    config.update_from_args(args)

    return config


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print(config)
