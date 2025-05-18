# Comprehensive Analysis of Overfitting in SSL_MAE

This document provides a detailed analysis of the overfitting issues in the SSL_MAE model and offers specific recommendations to address them.

## Problem Identification

The model is experiencing severe overfitting, as evidenced by:
- Validation accuracy consistently at or near 1.0 (100%)
- Validation loss at or near 0.0
- Training loss significantly higher than validation loss

This indicates that the model is not generalizing well and is likely memorizing the training data.

## Root Causes

After a thorough analysis of the codebase, I've identified several root causes:

### 1. Data Leakage Between Training and Validation Sets

The most critical issue is **data leakage** between training and validation sets. The current data splitting approach uses random splitting:

```python
# Split into training and validation sets
train_size = int(config.train_val_pct * len(file_paths))
train_file_paths = file_paths[:train_size]
val_file_paths = file_paths[train_size:]
```

This doesn't account for subject identity. Looking at the dataset structure:
- Bonafide images have filenames like `04376-1.jpg` (subject ID 04376)
- Morph images have filenames like `04376d103-vs-04379d116.jpg` (morphs of subjects 04376 and 04379)

With random splitting, images from the same subject can appear in both training and validation sets. This means the model is "memorizing" specific subjects rather than learning generalizable features for morph detection.

### 2. Implementation Issues with Subject-Based Splitting

The subject-based splitting implementation has import issues:
- The error message "Warning: subject_splitter module not available" indicates that the import is failing
- The import path is incorrect, preventing the subject-based splitting from being applied

### 3. Device Mismatch in LoRA Implementation

There's a device mismatch in the LoRA implementation:
- LoRA parameters (`lora_A` and `lora_B`) are initialized on CPU but not explicitly moved to the GPU
- This causes a runtime error: "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"

### 4. Complex Model Architecture

The model architecture is quite complex for a small dataset:
- Pre-trained ViT MAE backbone
- Optional projection head
- Multi-layer classifier
- Various loss functions (margin loss, reconstruction loss)
- LoRA adaptation

### 5. Class Weights Initialization

Class weights are initialized on CPU and not explicitly moved to the correct device during the forward pass.

## Implemented Fixes

I've implemented the following fixes to address these issues:

### 1. Fixed Subject Splitter Import

Updated the import logic in `data_loader.py` to try multiple import paths:

```python
try:
    # Try direct import first
    from subject_splitter import split_by_subjects
    SUBJECT_SPLITTING_AVAILABLE = True
except ImportError:
    try:
        # Try absolute import if direct import fails
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.subject_splitter import split_by_subjects
        SUBJECT_SPLITTING_AVAILABLE = True
    except ImportError:
        try:
            # Try relative import if absolute import fails
            from .subject_splitter import split_by_subjects
            SUBJECT_SPLITTING_AVAILABLE = True
        except ImportError:
            SUBJECT_SPLITTING_AVAILABLE = False
            print("Warning: subject_splitter module not available. Subject-based splitting will be disabled.")
```

### 2. Fixed Device Mismatch in LoRA

Updated the `forward` method in `LoRALayer` to ensure parameters are on the correct device:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Ensure lora_A and lora_B are on the same device as x
    if self.lora_A.device != x.device:
        self.lora_A = self.lora_A.to(x.device)
        self.lora_B = self.lora_B.to(x.device)
        
    return (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

### 3. Improved MarginLoss Implementation

Updated the `MarginLoss` class to handle device mismatch:

```python
def forward(self, logits, target):
    # ... existing code ...
    
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
```

## Additional Recommendations

Despite these fixes, more changes are needed to fully address the overfitting issue:

### 1. Enable Subject-Based Splitting

The most important recommendation is to **enable subject-based splitting**:

```bash
python src/train.py --override use_subject_splitting=True
```

This will ensure that images from the same subject are either all in the training set or all in the validation set, preventing data leakage.

### 2. Simplify the Model Architecture

Reduce model complexity to prevent overfitting:

```bash
python src/train.py --override use_projection_head=False --override use_improved_classifier=False
```

### 3. Increase Regularization

Add more regularization to prevent overfitting:

```bash
python src/train.py --override dropout_p=0.8 --override weight_decay=0.1
```

### 4. Disable Reconstruction Loss

The reconstruction loss might be contributing to overfitting:

```bash
python src/train.py --override recon_weight=0.0
```

### 5. Use Extreme Data Augmentation

Increase data augmentation to improve generalization:

```bash
python src/train.py --override use_advanced_aug=True --override advanced_aug_magnitude=10
```

### 6. Evaluate on External Datasets

Test the model on completely different datasets to assess true generalization.

## Conclusion

The overfitting issue in the SSL_MAE model is primarily due to data leakage between training and validation sets. By implementing subject-based splitting and the other recommendations above, you should see more realistic validation metrics and better generalization to unseen data.

The perfect validation accuracy you're currently seeing is misleading and indicates that your model is not learning generalizable features for morph detection. With these changes, you'll get a more accurate picture of your model's true performance.
