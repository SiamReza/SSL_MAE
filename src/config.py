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
        self.train_dataset = "StyleGAN_IWBF"  # Default training dataset
        self.test_datasets = ["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN_IWBF"] #["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN_IWBF"]
        self.train_val_pct = 0.85  # 85% for training, 15% for validation

        # Training Hyperparameters
        self.num_epochs = 15
        self.batch_size = 32
        self.learning_rate = 5e-5
        self.recon_weight = 1.0  # Weight for reconstruction loss (1 for MAE+classifier, 0 for supervised-only)
        self.lr_scheduler = "cosine"  # 'cosine' or 'none'
        self.eta_min = 0.0  # Minimum learning rate at the end of cosine schedule
        self.T_max = self.num_epochs  # Number of epochs over which to decay
        self.use_fixed_seed = True  # Whether to use a fixed seed (True) or random seed (False)
        self.seed = 42  # Seed value when use_fixed_seed is True

        # Regularization and Backbone Control
        self.freeze_strategy = "gradual_unfreeze"  # 'freeze_except_lastN', 'none', 'backbone_only', or 'gradual_unfreeze'
        self.warmup_epochs = 2  # Number of epochs to train only the head (for 'gradual_unfreeze')
        self.mid_epochs = 5  # Epoch to start unfreezing the full backbone (for 'gradual_unfreeze')
        self.freeze_lastN = 2  # Number of final ViT blocks to keep trainable

        self.dropout_p = 0.5  # Dropout probability
        self.use_mae_aug = True  # Whether to use MAE-specific augmentations
        self.mae_mask_ratio = 0.75  # Ratio of patches to mask in MAE (0.0 to 1.0)
        self.use_cls_aug = True  # Whether to use classifier augmentations

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

    args, _ = parser.parse_known_args()

    config = Config()
    config.update_from_args(args)

    return config


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print(config)
