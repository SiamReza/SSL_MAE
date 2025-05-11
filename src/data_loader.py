"""
Data loading module for the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection.
This module handles loading and preprocessing images from the dataset.
"""

import os
import glob
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class MorphDataset(Dataset):
    """Dataset class for morph detection."""

    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        transform=None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the dataset.

        Args:
            file_paths: List of image file paths
            labels: List of labels (0 for bonafide, 1 for morph)
            transform: Torchvision transforms to apply to images
            metadata: Optional list of metadata dictionaries for each sample
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.metadata = metadata if metadata is not None else [{}] * len(file_paths)

        assert len(self.file_paths) == len(self.labels) == len(self.metadata), \
            "File paths, labels, and metadata must have the same length"

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with image, label, and metadata
        """
        image_path = self.file_paths[idx]
        label = self.labels[idx]
        metadata = self.metadata[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float32),
            'metadata': metadata,
            'path': image_path
        }


def get_transforms(config, is_training: bool = True) -> transforms.Compose:
    """
    Get image transformations based on configuration.

    Args:
        config: Configuration object
        is_training: Whether to use training or testing transforms

    Returns:
        Composed transforms
    """
    # Basic transforms (always applied)
    transform_list = [
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # Add classifier augmentations if enabled and in training mode
    if is_training and config.use_cls_aug:
        transform_list = [
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.1), ratio=(0.9, 1.1),
                                         interpolation=transforms.InterpolationMode.LANCZOS),
        ] + transform_list

    # Note: MAE augmentations (masking patches) are not handled here
    # They are applied by the model during the forward pass, not in the data loader
    # The config.use_mae_aug flag will be used by the model to determine whether to apply masking

    return transforms.Compose(transform_list)


def get_file_paths(
    config,
    dataset_name: str,
    subset: str = "P2",
    is_training: bool = True
) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    """
    Get file paths, labels, and metadata for a dataset.

    Args:
        config: Configuration object
        dataset_name: Name of the morphing style dataset
        subset: Which subset to use ('P1' or 'P2')
        is_training: Whether this is for training/validation or testing

    Returns:
        Tuple of (file_paths, labels, metadata)
    """
    data_root = config.data_root

    # Get bonafide images
    bonafide_dir = os.path.join(data_root, "bonafide", dataset_name, subset)
    bonafide_paths = glob.glob(os.path.join(bonafide_dir, "*.jpg")) + glob.glob(os.path.join(bonafide_dir, "*.png"))
    bonafide_labels = [0] * len(bonafide_paths)
    bonafide_metadata = [{"type": "bonafide", "style": dataset_name} for _ in bonafide_paths]

    # Get morph images
    morph_dir = os.path.join(data_root, "morph", dataset_name, subset)
    morph_paths = glob.glob(os.path.join(morph_dir, "*.jpg")) + glob.glob(os.path.join(morph_dir, "*.png"))
    morph_labels = [1] * len(morph_paths)
    morph_metadata = [{"type": "morph", "style": dataset_name} for _ in morph_paths]

    # Combine and shuffle
    file_paths = bonafide_paths + morph_paths
    labels = bonafide_labels + morph_labels
    metadata = bonafide_metadata + morph_metadata

    # If this is for training/validation, shuffle the data
    if is_training:
        indices = list(range(len(file_paths)))
        np.random.shuffle(indices)
        file_paths = [file_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
        metadata = [metadata[i] for i in indices]

    return file_paths, labels, metadata


def create_data_loaders(config) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        config: Configuration object

    Returns:
        Dictionary with data loaders
    """
    # Get transforms
    train_transform = get_transforms(config, is_training=True)
    test_transform = get_transforms(config, is_training=False)

    # Get file paths for training dataset (P2 subset)
    train_dataset_name = config.train_dataset
    file_paths, labels, metadata = get_file_paths(
        config, train_dataset_name, subset="P2", is_training=True
    )

    # Split into training and validation sets
    train_size = int(config.train_val_pct * len(file_paths))
    val_size = len(file_paths) - train_size

    train_file_paths = file_paths[:train_size]
    train_labels = labels[:train_size]
    train_metadata = metadata[:train_size]

    val_file_paths = file_paths[train_size:]
    val_labels = labels[train_size:]
    val_metadata = metadata[train_size:]

    # Create datasets
    train_dataset = MorphDataset(
        train_file_paths, train_labels, transform=train_transform, metadata=train_metadata
    )

    val_dataset = MorphDataset(
        val_file_paths, val_labels, transform=test_transform, metadata=val_metadata
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Return dictionary with data loaders
    return {
        "train": train_loader,
        "val": val_loader
    }


def create_test_loaders(config, dataset_name: str) -> Dict[str, DataLoader]:
    """
    Create separate test data loaders for bonafide and morph images of a specific dataset.

    Args:
        config: Configuration object
        dataset_name: Name of the morphing style dataset

    Returns:
        Dictionary with separate data loaders for bonafide and morph images
    """
    # Get transforms
    test_transform = get_transforms(config, is_training=False)

    data_root = config.data_root
    subset = "P1"  # Always use P1 for testing

    # Get bonafide images
    bonafide_dir = os.path.join(data_root, "bonafide", dataset_name, subset)
    bonafide_paths = glob.glob(os.path.join(bonafide_dir, "*.jpg"))
    bonafide_labels = [0] * len(bonafide_paths)
    bonafide_metadata = [{"type": "bonafide", "style": dataset_name} for _ in bonafide_paths]

    # Get morph images
    morph_dir = os.path.join(data_root, "morph", dataset_name, subset)
    morph_paths = glob.glob(os.path.join(morph_dir, "*.jpg"))
    morph_labels = [1] * len(morph_paths)
    morph_metadata = [{"type": "morph", "style": dataset_name} for _ in morph_paths]

    # Create separate datasets
    bonafide_dataset = MorphDataset(
        bonafide_paths, bonafide_labels, transform=test_transform, metadata=bonafide_metadata
    )

    morph_dataset = MorphDataset(
        morph_paths, morph_labels, transform=test_transform, metadata=morph_metadata
    )

    # Create separate data loaders
    bonafide_loader = DataLoader(
        bonafide_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    morph_loader = DataLoader(
        morph_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return {
        "bonafide": bonafide_loader,
        "morph": morph_loader
    }


if __name__ == "__main__":
    # Example usage
    from config import get_config

    config = get_config()
    data_loaders = create_data_loaders(config)

    # Print dataset sizes
    print(f"Training set size: {len(data_loaders['train'].dataset)}")
    print(f"Validation set size: {len(data_loaders['val'].dataset)}")

    # Test creating test loaders
    test_loaders = create_test_loaders(config, "LMA")
    print(f"Test set size (LMA bonafide): {len(test_loaders['bonafide'].dataset)}")
    print(f"Test set size (LMA morph): {len(test_loaders['morph'].dataset)}")

    # Get a sample batch
    batch = next(iter(data_loaders['train']))
    print(f"Batch shape: {batch['image'].shape}")
    print(f"Labels shape: {batch['label'].shape}")
