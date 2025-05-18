"""
Testing script for the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection.
This script evaluates a trained model on multiple morphing techniques and generates evaluation metrics and visualizations.
"""

import os
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import get_config
from data_loader import create_test_loaders
from model import MorphDetector
from utils import (
    calculate_metrics, plot_roc_curve, plot_precision_recall_curve,
    plot_score_distributions, plot_confusion_matrix, save_results_to_csv,
    create_results_summary
)


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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Test Morph Detection Model')
    parser.add_argument('--model_path', type=str, required=False,
                        help='Path to the trained model')
    parser.add_argument('--datasets', type=str, nargs='+',
                        help='List of datasets to evaluate on')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--override', action='append',
                        help='Override configuration parameters, format: key=value')

    return parser.parse_args()


def load_model(config, model_path=None):
    """
    Load a trained model.

    Args:
        config: Configuration object
        model_path: Path to the trained model (if None, use default path)

    Returns:
        Loaded model
    """
    # Initialize model
    model = MorphDetector(config)

    # Determine model path
    if model_path is None:
        model_path = os.path.join(config.models_dir, config.final_model_name.format(config.train_dataset))

    # Load model weights
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    return model


def evaluate_dataset(model, test_loaders, dataset_name, config, threshold=0.5):
    """
    Evaluate model on a specific dataset.

    Args:
        model: The model to evaluate
        test_loaders: Dictionary with data loaders for bonafide and morph samples
        dataset_name: Name of the dataset
        config: Configuration object
        threshold: Classification threshold

    Returns:
        Dictionary with evaluation results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Initialize lists to store results
    all_labels = []
    all_scores = []

    # Process bonafide samples
    print(f"Evaluating bonafide samples for {dataset_name}...")
    bonafide_scores = []
    with torch.no_grad():
        for batch in tqdm(test_loaders['bonafide']):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            scores = outputs['scores'].squeeze().cpu().numpy()

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores)
            bonafide_scores.extend(scores)

    # Process morph samples
    print(f"Evaluating morph samples for {dataset_name}...")
    morph_scores = []
    with torch.no_grad():
        for batch in tqdm(test_loaders['morph']):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            scores = outputs['scores'].squeeze().cpu().numpy()

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores)
            morph_scores.extend(scores)

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    bonafide_scores = np.array(bonafide_scores)
    morph_scores = np.array(morph_scores)

    # Apply threshold to get predicted labels
    all_preds = (all_scores > threshold).astype(int)

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_scores)

    # Create output directory for plots
    plots_dir = os.path.join(config.plots_dir, dataset_name)
    os.makedirs(plots_dir, exist_ok=True)

    # Generate plots
    plot_roc_curve(
        all_labels, all_scores, dataset_name, config,
        save_path=os.path.join(plots_dir, f"{dataset_name}_roc.png")
    )

    plot_precision_recall_curve(
        all_labels, all_scores, dataset_name, config,
        save_path=os.path.join(plots_dir, f"{dataset_name}_pr.png")
    )

    plot_score_distributions(
        bonafide_scores, morph_scores, dataset_name, config,
        save_path=os.path.join(plots_dir, f"{dataset_name}_scores.png")
    )

    plot_confusion_matrix(
        all_labels, all_preds, dataset_name, config,
        save_path=os.path.join(plots_dir, f"{dataset_name}_cm.png")
    )

    return metrics


def test():
    """Main testing function."""
    # Parse command-line arguments
    args = parse_args()

    # Load configuration
    config = get_config()

    # Override configuration with command-line arguments
    if args.override:
        config.update_from_args(args)

    # Set seed for reproducibility
    seed = set_seed(config.seed, config.use_fixed_seed)
    seed_type = "fixed" if config.use_fixed_seed else "random"
    print(f"Using {'fixed' if config.use_fixed_seed else 'random'} seed: {seed}")

    # Determine which datasets to evaluate
    datasets = args.datasets if args.datasets else config.test_datasets

    # Load model
    model = load_model(config, args.model_path)

    # Create output directories
    os.makedirs(config.plots_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)

    # Evaluate on each dataset
    results = {}
    for dataset_name in datasets:
        print(f"\nEvaluating on {dataset_name} dataset...")

        # Create test loaders
        test_loaders = create_test_loaders(config, dataset_name)

        # Evaluate
        metrics = evaluate_dataset(model, test_loaders, dataset_name, config, args.threshold)

        # Store results
        results[dataset_name] = metrics

        # Print metrics
        print(f"\nResults for {dataset_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Save results to CSV
    model_name = os.path.basename(args.model_path).split('.')[0] if args.model_path else f"{config.train_dataset}_morphdetector"
    csv_path = save_results_to_csv(results, config, model_name, seed, seed_type)
    print(f"\nResults saved to {csv_path}")

    # Create and print summary
    summary = create_results_summary(results, model_name, seed, seed_type)
    summary_path = os.path.join(config.logs_dir, f"{model_name}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)

    print("\nResults Summary:")
    print(summary)
    print(f"Summary saved to {summary_path}")

    # Create comparative plots across datasets
    create_comparative_plots(results, config, model_name)


def create_comparative_plots(results, config, model_name):
    """
    Create comparative plots across all datasets.

    Args:
        results: Dictionary with evaluation results for each dataset
        config: Configuration object
        model_name: Name of the model
    """
    # Extract metrics for comparison
    datasets = list(results.keys())
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'eer']

    # Create bar plots for each metric
    for metric in metrics_to_plot:
        values = [results[dataset][metric] for dataset in datasets]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(datasets, values, color='skyblue')

        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{value:.3f}',
                ha='center', va='bottom', rotation=0
            )

        plt.xlabel('Dataset')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Across Datasets')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(config.plots_dir, f"{model_name}_{metric}_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Create BPCER comparison plot
    bpcer_metrics = ['bpcer_5', 'bpcer_10', 'bpcer_20']
    bpcer_labels = ['BPCER@APCER=5%', 'BPCER@APCER=10%', 'BPCER@APCER=20%']

    plt.figure(figsize=(14, 8))

    x = np.arange(len(datasets))
    width = 0.25

    for i, (metric, label) in enumerate(zip(bpcer_metrics, bpcer_labels)):
        values = [results[dataset][metric] for dataset in datasets]
        plt.bar(x + i*width - width, values, width, label=label)

    plt.xlabel('Dataset')
    plt.ylabel('BPCER Value')
    plt.title('BPCER Metrics Across Datasets')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(config.plots_dir, f"{model_name}_bpcer_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    test()
