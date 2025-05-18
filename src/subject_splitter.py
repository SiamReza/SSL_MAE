"""
Subject-based splitting module for the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection.
This module handles splitting data based on subject IDs to prevent data leakage between training and validation sets.
"""

import os
import re
from typing import Dict, List, Tuple, Any, Set
import numpy as np
from sklearn.model_selection import train_test_split


def extract_subject_id(filename: str) -> str:
    """
    Extract subject ID from a filename.
    
    Args:
        filename: Filename to extract subject ID from
        
    Returns:
        Subject ID as a string
    """
    basename = os.path.basename(filename)
    
    # Check if it's a morph image (contains "-vs-")
    if "-vs-" in basename:
        # For morph images, extract both subject IDs
        # Format: {subject1}d{suffix1}-vs-{subject2}d{suffix2}.jpg
        parts = basename.split("-vs-")
        subject1 = parts[0].split("d")[0]  # Extract first subject ID
        subject2 = parts[1].split("d")[0]  # Extract second subject ID
        return f"{subject1}_{subject2}"  # Return combined ID to track both subjects
    else:
        # For bonafide images, extract the subject ID
        # Format: {subject}-{sequence}.jpg
        subject = basename.split("-")[0]
        return subject


def get_all_subject_ids(file_paths: List[str]) -> Set[str]:
    """
    Get all unique subject IDs from a list of file paths.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        Set of unique subject IDs
    """
    # Extract subject IDs from all file paths
    subject_ids = set()
    for path in file_paths:
        subject_id = extract_subject_id(path)
        
        # For morph images, add both individual subject IDs
        if "_" in subject_id:
            subj1, subj2 = subject_id.split("_")
            subject_ids.add(subj1)
            subject_ids.add(subj2)
        else:
            subject_ids.add(subject_id)
    
    return subject_ids


def split_by_subjects(
    file_paths: List[str],
    labels: List[int],
    metadata: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    random_state: int = 42
) -> Tuple[List[str], List[int], List[Dict[str, Any]], List[str], List[int], List[Dict[str, Any]]]:
    """
    Split data into training and validation sets based on subject IDs.
    
    Args:
        file_paths: List of file paths
        labels: List of labels
        metadata: List of metadata
        train_ratio: Ratio of subjects to use for training
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_file_paths, train_labels, train_metadata, val_file_paths, val_labels, val_metadata)
    """
    # Get all unique subject IDs
    all_subject_ids = get_all_subject_ids(file_paths)
    
    # Convert to list for splitting
    all_subject_ids = list(all_subject_ids)
    
    # Split subjects into training and validation sets
    train_subjects, val_subjects = train_test_split(
        all_subject_ids,
        train_size=train_ratio,
        random_state=random_state
    )
    
    # Convert to sets for faster lookup
    train_subjects = set(train_subjects)
    val_subjects = set(val_subjects)
    
    # Initialize lists for training and validation data
    train_file_paths = []
    train_labels = []
    train_metadata = []
    val_file_paths = []
    val_labels = []
    val_metadata = []
    
    # Assign data to training or validation based on subject IDs
    for i, path in enumerate(file_paths):
        subject_id = extract_subject_id(path)
        
        # For morph images, check if both subjects are in the same set
        if "_" in subject_id:
            subj1, subj2 = subject_id.split("_")
            
            # If both subjects are in training set, assign to training
            if subj1 in train_subjects and subj2 in train_subjects:
                train_file_paths.append(path)
                train_labels.append(labels[i])
                train_metadata.append(metadata[i])
            # If both subjects are in validation set, assign to validation
            elif subj1 in val_subjects and subj2 in val_subjects:
                val_file_paths.append(path)
                val_labels.append(labels[i])
                val_metadata.append(metadata[i])
            # If subjects are split between sets, assign to validation
            # This is a conservative approach to prevent data leakage
            else:
                val_file_paths.append(path)
                val_labels.append(labels[i])
                val_metadata.append(metadata[i])
        else:
            # For bonafide images, check if subject is in training or validation set
            if subject_id in train_subjects:
                train_file_paths.append(path)
                train_labels.append(labels[i])
                train_metadata.append(metadata[i])
            else:
                val_file_paths.append(path)
                val_labels.append(labels[i])
                val_metadata.append(metadata[i])
    
    # Print statistics
    print(f"Subject-based splitting statistics:")
    print(f"  Total subjects: {len(all_subject_ids)}")
    print(f"  Training subjects: {len(train_subjects)}")
    print(f"  Validation subjects: {len(val_subjects)}")
    print(f"  Training samples: {len(train_file_paths)}")
    print(f"  Validation samples: {len(val_file_paths)}")
    print(f"  Training class distribution: {np.bincount(train_labels)}")
    print(f"  Validation class distribution: {np.bincount(val_labels)}")
    
    return train_file_paths, train_labels, train_metadata, val_file_paths, val_labels, val_metadata


if __name__ == "__main__":
    # Test the subject-based splitting
    import glob
    from config import get_config
    
    config = get_config()
    data_root = config.data_root
    dataset_name = config.train_dataset
    subset = "train"
    
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
    
    # Combine
    file_paths = bonafide_paths + morph_paths
    labels = bonafide_labels + morph_labels
    metadata = bonafide_metadata + morph_metadata
    
    # Split by subjects
    train_file_paths, train_labels, train_metadata, val_file_paths, val_labels, val_metadata = split_by_subjects(
        file_paths, labels, metadata, train_ratio=config.train_val_pct, random_state=config.seed
    )
    
    # Print some examples
    print("\nTraining examples:")
    for i in range(min(5, len(train_file_paths))):
        print(f"  {os.path.basename(train_file_paths[i])}: {train_labels[i]}")
    
    print("\nValidation examples:")
    for i in range(min(5, len(val_file_paths))):
        print(f"  {os.path.basename(val_file_paths[i])}: {val_labels[i]}")
