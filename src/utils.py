"""
Utility functions for the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection.
This module contains functions for evaluation metrics, visualization, and result analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix, classification_report
)
import torch
from typing import Dict, List, Tuple, Optional, Union, Any


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for binary classification.

    Args:
        y_true: Ground truth labels (0 for bonafide, 1 for morph)
        y_pred: Predicted labels (0 for bonafide, 1 for morph)
        y_score: Prediction scores (probabilities) for the morph class

    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate AUC
    auc = roc_auc_score(y_true, y_score)

    # Calculate Average Precision (AP)
    ap = average_precision_score(y_true, y_score)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate Equal Error Rate (EER)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    # Calculate BPCER (Bonafide Presentation Classification Error Rate) at different APCER thresholds
    # APCER: Attack Presentation Classification Error Rate (proportion of attack samples incorrectly classified as bonafide)
    # BPCER: Bonafide Presentation Classification Error Rate (proportion of bonafide samples incorrectly classified as attack)

    # Get bonafide and attack scores
    bonafide_scores = y_score[y_true == 0]
    attack_scores = y_score[y_true == 1]

    # Calculate BPCER at APCER = 5%
    if len(attack_scores) > 0:
        threshold_5 = np.percentile(attack_scores, 5)
        bpcer_5 = (bonafide_scores >= threshold_5).mean()
    else:
        bpcer_5 = 0

    # Calculate BPCER at APCER = 10%
    if len(attack_scores) > 0:
        threshold_10 = np.percentile(attack_scores, 10)
        bpcer_10 = (bonafide_scores >= threshold_10).mean()
    else:
        bpcer_10 = 0

    # Calculate BPCER at APCER = 20%
    if len(attack_scores) > 0:
        threshold_20 = np.percentile(attack_scores, 20)
        bpcer_20 = (bonafide_scores >= threshold_20).mean()
    else:
        bpcer_20 = 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'auc': auc,
        'ap': ap,
        'eer': eer,
        'bpcer_5': bpcer_5,
        'bpcer_10': bpcer_10,
        'bpcer_20': bpcer_20
    }


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, dataset_name: str, config, save_path: Optional[str] = None) -> None:
    """
    Plot ROC curve for binary classification.

    Args:
        y_true: Ground truth labels (0 for bonafide, 1 for morph)
        y_score: Prediction scores (probabilities) for the morph class
        dataset_name: Name of the dataset
        config: Configuration object
        save_path: Path to save the plot (if None, plot is displayed)
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray, dataset_name: str, config, save_path: Optional[str] = None) -> None:
    """
    Plot Precision-Recall curve for binary classification.

    Args:
        y_true: Ground truth labels (0 for bonafide, 1 for morph)
        y_score: Prediction scores (probabilities) for the morph class
        dataset_name: Name of the dataset
        config: Configuration object
        save_path: Path to save the plot (if None, plot is displayed)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name}')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_score_distributions(bonafide_scores: np.ndarray, morph_scores: np.ndarray, dataset_name: str, config, save_path: Optional[str] = None) -> None:
    """
    Plot score distributions for bonafide and morph samples.

    Args:
        bonafide_scores: Prediction scores for bonafide samples
        morph_scores: Prediction scores for morph samples
        dataset_name: Name of the dataset
        config: Configuration object
        save_path: Path to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=(10, 6))

    # Plot histograms
    sns.histplot(bonafide_scores, color='green', alpha=0.5, label='Bonafide', kde=True, bins=30)
    sns.histplot(morph_scores, color='red', alpha=0.5, label='Morph', kde=True, bins=30)

    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title(f'Score Distributions - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str, config, save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix for binary classification.

    Args:
        y_true: Ground truth labels (0 for bonafide, 1 for morph)
        y_pred: Predicted labels (0 for bonafide, 1 for morph)
        dataset_name: Name of the dataset
        config: Configuration object
        save_path: Path to save the plot (if None, plot is displayed)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Bonafide', 'Morph'],
                yticklabels=['Bonafide', 'Morph'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {dataset_name}')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_results_to_csv(results: Dict[str, Dict[str, float]], config, model_name: str, seed: int = None, seed_type: str = None) -> str:
    """
    Save evaluation results to CSV file.

    Args:
        results: Dictionary with evaluation results for each dataset
        config: Configuration object
        model_name: Name of the model
        seed: Seed value used for reproducibility
        seed_type: Type of seed (fixed or random)

    Returns:
        Path to the saved CSV file
    """
    # Create DataFrame from results
    data = []
    for dataset_name, metrics in results.items():
        row = {'dataset': dataset_name}
        if seed is not None:
            row['seed'] = seed
        if seed_type is not None:
            row['seed_type'] = seed_type
        row.update(metrics)
        data.append(row)

    df = pd.DataFrame(data)

    # Save to CSV
    os.makedirs(config.logs_dir, exist_ok=True)
    csv_path = os.path.join(config.logs_dir, f"{model_name}_results.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


def create_results_summary(results: Dict[str, Dict[str, float]], model_name: str, seed: int = None, seed_type: str = None) -> str:
    """
    Create a text summary of evaluation results.

    Args:
        results: Dictionary with evaluation results for each dataset
        model_name: Name of the model
        seed: Seed value used for reproducibility
        seed_type: Type of seed (fixed or random)

    Returns:
        Text summary of results
    """
    summary = f"Results Summary for {model_name}\n"
    summary += "=" * 80 + "\n\n"

    # Add seed information if provided
    if seed is not None:
        summary += f"Seed: {seed} ({seed_type})\n\n"

    # Calculate average metrics across all datasets
    all_metrics = {}
    for dataset_name, metrics in results.items():
        for metric_name, value in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    avg_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}

    # Add average metrics to summary
    summary += "Average Metrics Across All Datasets:\n"
    summary += "-" * 40 + "\n"
    for metric, value in avg_metrics.items():
        summary += f"{metric}: {value:.4f}\n"

    summary += "\n"

    # Add per-dataset metrics to summary
    summary += "Per-Dataset Metrics:\n"
    summary += "-" * 40 + "\n"

    for dataset_name, metrics in results.items():
        summary += f"\n{dataset_name}:\n"
        for metric, value in metrics.items():
            summary += f"  {metric}: {value:.4f}\n"

    return summary
