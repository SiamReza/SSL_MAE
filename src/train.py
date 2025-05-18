"""
Training script for the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection.
This script handles the complete training workflow, including setup, training loop, validation, logging, and model saving.
"""

import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from config import get_config
from data_loader import create_data_loaders, MorphDataset, get_transforms
from model import MorphDetector


def set_seed(seed=None, use_fixed_seed=True):
    """
    Set random seed for reproducibility.

    Args:
        seed: Seed value to use if use_fixed_seed is True
        use_fixed_seed: Whether to use a fixed seed or generate a random one

    Returns:
        The seed value used
    """
    if use_fixed_seed:
        # Use the provided seed
        random_seed = seed
    else:
        # Generate a random seed
        random_seed = random.randint(0, 10000)

    # Set seeds for Python, NumPy, and PyTorch
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        # For full reproducibility (may affect performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return random_seed


def create_output_dirs(config):
    """
    Create output directories for models, logs, and plots.

    Args:
        config: Configuration object
    """
    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)


def initialize_csv_logger(config, seed, seed_type):
    """
    Initialize CSV logger for training metrics.

    Args:
        config: Configuration object
        seed: Seed value used
        seed_type: Whether the seed was fixed or random

    Returns:
        Path to the CSV log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.logs_dir, f"train_{config.train_dataset}_{timestamp}.csv")

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Add fold column if using cross-validation
        if config.use_cross_validation:
            header = ['seed', 'seed_type', 'epoch', 'fold'] + ['train_loss', 'val_loss', 'val_acc', 'learning_rate']
        else:
            header = ['seed', 'seed_type', 'epoch'] + ['train_loss', 'val_loss', 'val_acc', 'learning_rate']
        writer.writerow(header)

    return log_file


def log_metrics(log_file, metrics, seed, epoch=None, fold_idx=None):
    """
    Log metrics to CSV file.

    Args:
        log_file: Path to the CSV log file
        metrics: Dictionary of metrics to log
        seed: Seed value used
        epoch: Current epoch number
        fold_idx: Fold index for cross-validation
    """
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [seed]

        if epoch is None:
            # This is the initial row with seed type
            row.append(metrics['seed_type'])
            # Add placeholder values for other metrics
            row.extend([''] * (len(metrics) - 1))
        else:
            # Regular metrics row
            row.append('')  # Empty seed_type column
            row.append(epoch)

            # Add fold index if using cross-validation
            if fold_idx is not None:
                row.append(fold_idx + 1)  # 1-based fold index for display
            else:
                row.append('')  # Empty fold column

            for metric in ['train_loss', 'val_loss', 'val_acc', 'learning_rate']:
                row.append(metrics.get(metric, ''))

        writer.writerow(row)


def validate(model, val_loader, device):
    """
    Evaluate the model on the validation set.

    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation data
        device: Device to run evaluation on

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, labels)
            val_loss += outputs['loss'].item()

            predicted = (outputs['scores'].squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / total

    return {
        'val_loss': val_loss,
        'val_acc': val_acc
    }


def plot_metrics(metrics, config):
    """
    Plot training and validation metrics.

    Args:
        metrics: Dictionary of metrics
        config: Configuration object
    """
    epochs = range(1, len(metrics['train_loss']) + 1)

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.plots_dir, f"{config.train_dataset}_loss.png"))

    # Plot accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics['val_acc'], 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.plots_dir, f"{config.train_dataset}_accuracy.png"))


def train_fold(train_dataset, val_dataset, config, device, log_file, seed, fold_idx=None):
    """
    Train a model on a single fold.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration object
        device: Device to run training on
        log_file: Path to log file
        seed: Random seed
        fold_idx: Fold index (None for regular training)

    Returns:
        Trained model and metrics
    """
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Initialize model
    model = MorphDetector(config)
    model.to(device)

    # Initialize optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Initialize learning rate scheduler
    if config.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min
        )
    else:
        scheduler = None

    # Initialize metrics tracking
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    # Initialize early stopping variables if enabled
    if config.use_early_stopping:
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        fold_str = f" (Fold {fold_idx+1}/{config.n_folds})" if fold_idx is not None else ""
        print(f"Early stopping enabled with patience {config.patience}{fold_str}")

    # Training loop
    fold_str = f" (Fold {fold_idx+1}/{config.n_folds})" if fold_idx is not None else ""
    print(f"Starting training for {config.num_epochs} epochs{fold_str}...")

    for epoch in range(1, config.num_epochs + 1):
        # Apply freezing strategy for gradual unfreezing
        if config.freeze_strategy == 'gradual_unfreeze':
            if epoch == config.warmup_epochs:
                print(f"Epoch {epoch}: Unfreezing last {config.freeze_lastN} blocks")
                model.unfreeze_last_n_blocks(config.freeze_lastN)
            elif epoch == config.mid_epochs:
                print(f"Epoch {epoch}: Unfreezing all layers")
                model.unfreeze_all()

        # Training phase
        model.train()
        train_loss = 0.0

        # Progress bar for training
        epoch_desc = f"Epoch {epoch}/{config.num_epochs}{fold_str} [Train]"
        train_pbar = tqdm(train_loader, desc=epoch_desc)
        for batch in train_pbar:
            # Get data
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, labels)
            loss = outputs['loss']

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        # Calculate average training loss
        train_loss /= len(train_loader)

        # Validation phase
        val_metrics = validate(model, val_loader, device)

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        # Update metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_metrics['val_loss'])
        metrics['val_acc'].append(val_metrics['val_acc'])
        metrics['learning_rate'].append(current_lr)

        # Log metrics
        epoch_metrics = {
            'train_loss': train_loss,
            'val_loss': val_metrics['val_loss'],
            'val_acc': val_metrics['val_acc'],
            'learning_rate': current_lr
        }

        # Add fold information to log if using cross-validation
        if fold_idx is not None:
            log_metrics(log_file, epoch_metrics, seed, epoch, fold_idx)
        else:
            log_metrics(log_file, epoch_metrics, seed, epoch)

        # Print progress
        print(f"Epoch {epoch}/{config.num_epochs}{fold_str} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_metrics['val_loss']:.4f}, "
              f"Val Acc: {val_metrics['val_acc']:.4f}, "
              f"LR: {current_lr:.6f}")

        # Early stopping check
        if config.use_early_stopping:
            if val_metrics['val_loss'] < best_val_loss - config.min_delta:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"Validation loss improved to {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience: {patience_counter}/{config.patience}")
                if patience_counter >= config.patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    # Restore best model
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

    return model, metrics


def train():
    """Main training function."""
    # Load configuration
    config = get_config()

    # Set seed for reproducibility
    seed = set_seed(config.seed, config.use_fixed_seed)
    seed_type = "fixed" if config.use_fixed_seed else "random"
    print(f"Using {'fixed' if config.use_fixed_seed else 'random'} seed: {seed}")

    # Create output directories
    create_output_dirs(config)

    # Initialize CSV logger
    log_file = initialize_csv_logger(config, seed, seed_type)
    log_metrics(log_file, {'seed_type': seed_type}, seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data for training
    from data_loader import create_data_loaders, get_file_paths, MorphDataset, get_transforms

    if config.use_cross_validation:
        print(f"Using {config.n_folds}-fold cross-validation")

        # Get file paths and create dataset
        train_dataset_name = config.train_dataset
        file_paths, labels, metadata = get_file_paths(
            config, train_dataset_name, subset="train", is_training=True
        )

        # Create transforms
        train_transform = get_transforms(config, is_training=True)
        test_transform = get_transforms(config, is_training=False)

        # Create full dataset
        full_dataset = MorphDataset(
            file_paths, labels, transform=train_transform, metadata=metadata
        )

        # Initialize k-fold cross-validation
        kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=seed)

        # Initialize metrics for each fold
        fold_metrics = []

        # Train on each fold
        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(range(len(full_dataset)))):
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx+1}/{config.n_folds}")
            print(f"{'='*50}")

            # Create train and validation datasets for this fold
            train_subset = Subset(full_dataset, train_indices)

            # Create a separate validation dataset with test transforms
            val_file_paths = [file_paths[i] for i in val_indices]
            val_labels = [labels[i] for i in val_indices]
            val_metadata = [metadata[i] for i in val_indices]
            val_dataset = MorphDataset(
                val_file_paths, val_labels, transform=test_transform, metadata=val_metadata
            )

            # Train on this fold
            model, fold_metric = train_fold(
                train_subset, val_dataset, config, device, log_file, seed, fold_idx
            )

            # Save fold model
            fold_model_path = os.path.join(
                config.models_dir,
                f"{config.train_dataset}_fold{fold_idx+1}_morphdetector.pt"
            )
            torch.save(model.state_dict(), fold_model_path)
            print(f"Fold {fold_idx+1} model saved to {fold_model_path}")

            # Store metrics for this fold
            fold_metrics.append(fold_metric)

        # Calculate average metrics across folds
        avg_metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        # Find the maximum number of epochs across all folds
        max_epochs = max(len(metrics['train_loss']) for metrics in fold_metrics)

        # Pad metrics to the same length
        for metrics in fold_metrics:
            for key in avg_metrics.keys():
                # Pad with the last value if early stopping was triggered
                if len(metrics[key]) < max_epochs:
                    metrics[key].extend([metrics[key][-1]] * (max_epochs - len(metrics[key])))

        # Calculate average for each epoch
        for epoch in range(max_epochs):
            for key in avg_metrics.keys():
                avg_metrics[key].append(
                    sum(metrics[key][epoch] for metrics in fold_metrics) / config.n_folds
                )

        # Plot average metrics
        plot_metrics(avg_metrics, config)
        print(f"Cross-validation completed. Metrics saved to {log_file}")

    else:
        # Regular training without cross-validation
        # Create data loaders
        data_loaders = create_data_loaders(config)
        train_dataset = data_loaders['train'].dataset
        val_dataset = data_loaders['val'].dataset

        # Train the model
        model, metrics = train_fold(
            train_dataset, val_dataset, config, device, log_file, seed
        )

        # Save final model
        model_path = os.path.join(config.models_dir, config.final_model_name.format(config.train_dataset))
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Plot metrics
        plot_metrics(metrics, config)
        print(f"Training completed. Metrics saved to {log_file}")


if __name__ == "__main__":
    train()
