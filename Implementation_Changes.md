# Implementation Changes Documentation

This document outlines the changes made to the SSL_MAE codebase to address the identified issues and improve the model's performance, efficiency, and generalization capabilities.

## Overview of Changes

1. **CLI Flag for Mask Ratio**
2. **Separate Mask Generation Utility**
3. **Learned Mask Token in Decoder**
4. **Separate Positional Embeddings**
5. **Projection Head for Linear Probing**
6. **Improved Classification Head**
7. **Margin-Based Loss**
8. **Parameter-Efficient Fine-Tuning with LoRA**
9. **Balanced Data Loader**
10. **Advanced Augmentations**

## Configuration Options

All new features can be enabled or disabled through the configuration in `src/config.py`. The following options have been added:

```python
# Augmentation options
self.use_advanced_aug = False  # Whether to use advanced augmentations
self.advanced_aug_magnitude = 9  # Magnitude for RandAugment (1-10)
self.advanced_aug_num_ops = 2  # Number of operations for RandAugment

# Data loading options
self.use_balanced_sampler = False  # Whether to use balanced sampling

# Model architecture options
self.use_projection_head = False  # Whether to use projection head
self.proj_dim = 256  # Projection dimension
self.proj_hidden_dim = None  # Hidden dimension for projection head

self.use_improved_classifier = False  # Whether to use improved classifier
self.cls_hidden_dim = None  # Hidden dimension for classifier

# Loss function options
self.use_margin_loss = False  # Whether to use margin-based loss
self.margin = 0.3  # Margin for margin-based loss
self.class_weights = [1.0, 1.0]  # Class weights [bonafide, morph]

# MAE-specific options
self.use_learned_mask_token = False  # Whether to use learned mask token
self.use_separate_pos_embed = False  # Whether to use separate pos embeddings
self.decoder_depth = 4  # Number of transformer layers in decoder

# LoRA options
self.use_lora = False  # Whether to use LoRA
self.lora_r = 4  # Rank for LoRA adaptation
self.lora_alpha = 32  # Alpha parameter for LoRA
self.lora_dropout = 0.1  # Dropout probability for LoRA layers
self.lora_target_modules = ["query", "key", "value"]  # Target modules
```

## Detailed Explanation of Changes

### 1. CLI Flag for Mask Ratio

**What Changed:**
- Added a command-line argument `--mask_ratio` to override the default mask ratio.

**How to Use:**
```bash
python src/main.py --mask_ratio 0.6
```

**Benefits:**
- Enables quick experimentation with different mask ratios without modifying code
- Makes it easier to perform ablation studies

### 2. Separate Mask Generation Utility

**What Changed:**
- Created a new file `src/mask_utils.py` with dedicated masking functions
- Implemented `random_masking()` and `apply_mask()` functions

**How to Use:**
```python
from mask_utils import random_masking, apply_mask

# Apply masking to a tensor
x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.75)
```

**Benefits:**
- Decouples masking logic from the model implementation
- Makes the code more modular, testable, and reusable
- Allows for experimentation with different masking strategies

### 3. Learned Mask Token in Decoder

**What Changed:**
- Added a learnable mask token parameter to the model
- Implemented a decoder that uses this mask token to reconstruct masked patches

**How to Enable:**
```python
config.use_learned_mask_token = True
```

**Benefits:**
- Follows the original MAE paper's approach for better reconstruction
- Improves the model's ability to learn meaningful representations
- Enables proper reconstruction of masked patches

### 4. Separate Positional Embeddings

**What Changed:**
- Added separate positional embeddings for encoder and decoder
- Implemented logic to use the appropriate embeddings in each forward pass

**How to Enable:**
```python
config.use_separate_pos_embed = True
```

**Benefits:**
- Allows each component to learn optimal positional representations
- Improves both encoding and decoding performance
- Follows the approach used in the original MAE paper

### 5. Projection Head for Linear Probing

**What Changed:**
- Created a new file `src/heads.py` with a `ProjectionHead` class
- Added this projection head to the model to create a better representation space

**How to Enable:**
```python
config.use_projection_head = True
config.proj_dim = 256  # Optional: customize projection dimension
```

**Benefits:**
- Creates a better representation space for downstream tasks
- Follows the approach of contrastive learning methods like SimCLR
- Improves linear evaluation performance

### 6. Improved Classification Head

**What Changed:**
- Created a new file `src/classifier.py` with a `MorphClassifier` class
- Replaced the simple linear classifier with a more sophisticated MLP

**How to Enable:**
```python
config.use_improved_classifier = True
config.cls_hidden_dim = 512  # Optional: customize hidden dimension
```

**Benefits:**
- Increases model capacity for the classification task
- Can capture more complex relationships in the data
- Potentially improves classification performance

### 7. Margin-Based Loss

**What Changed:**
- Added support for MarginLoss from torchmetrics
- Implemented logic to use this loss instead of binary cross-entropy

**How to Enable:**
```python
config.use_margin_loss = True
config.margin = 0.3  # Optional: customize margin
config.class_weights = [1.0, 1.0]  # Optional: customize class weights
```

**Benefits:**
- Enforces a margin between class embeddings
- Increases robustness by pushing classes further apart
- Can improve generalization to unseen data

### 8. Parameter-Efficient Fine-Tuning with LoRA

**What Changed:**
- Created a new file `src/lora.py` with LoRA implementation
- Added support for applying LoRA adapters to attention layers

**How to Enable:**
```python
config.use_lora = True
config.lora_r = 4  # Optional: customize rank
config.lora_alpha = 32  # Optional: customize alpha
```

**How to Run Training with LoRA:**
```bash
# First, freeze the backbone
python src/main.py --override use_lora=True --override freeze_backbone=True

# Fine-tune with LoRA
python src/main.py --override use_lora=True --override freeze_backbone=True --override learning_rate=1e-4
```

**Differences from Previous Training Method:**
- **Parameter Efficiency**: LoRA only trains a small number of adapter parameters (typically <1% of the total model size) instead of fine-tuning the entire model or specific layers.
- **Training Process**: The backbone model remains completely frozen, and only the LoRA adapters and classification head are updated during training.
- **Checkpoint Size**: The saved checkpoints will be much smaller as they only need to store the LoRA adapter weights and classification head.
- **Training Speed**: Training is typically faster as fewer parameters need to be updated, and less memory is required for gradients.
- **Convergence**: LoRA often converges faster than full fine-tuning, requiring fewer epochs to reach optimal performance.

**Files Generated:**
- The model checkpoint saved at `checkpoints/[timestamp]_model.pt` will contain:
  - The frozen backbone weights (unchanged from pre-training)
  - The LoRA adapter weights (small matrices for each attention layer)
  - The classification head weights
- Training logs will be saved at `logs/[timestamp]_training.log`
- Evaluation results will be saved at `results/[timestamp]_eval.json`

**How to Test LoRA Models:**
```bash
# Test a LoRA-trained model
python src/test.py --model_path checkpoints/[timestamp]_model.pt --override use_lora=True
```

**Benefits:**
- Reduces the number of trainable parameters (from millions to thousands)
- Prevents overfitting, especially with limited data
- Makes training more efficient (faster and less memory-intensive)
- Preserves pre-trained knowledge while adapting to the target task
- Enables effective transfer learning even with small datasets

### 9. Balanced Data Loader

**What Changed:**
- Added a `make_balanced_sampler()` function to `data_loader.py`
- Modified the data loader to use this sampler when enabled

**How to Enable:**
```python
config.use_balanced_sampler = True
```

**Benefits:**
- Ensures balanced class representation during training
- Prevents bias towards the majority class
- Improves performance on minority classes

### 10. Advanced Augmentations

**What Changed:**
- Added support for advanced augmentations using albumentations
- Implemented a more sophisticated augmentation pipeline

**How to Enable:**
```python
config.use_advanced_aug = True
config.advanced_aug_magnitude = 9  # Optional: customize magnitude
config.advanced_aug_num_ops = 2  # Optional: customize number of operations
```

**Benefits:**
- Improves generalization across different domains
- Makes the model more robust to variations in capture devices
- Can lead to better performance on unseen data

## Required Dependencies

To use all the new features, you'll need to install the following dependencies:

```bash
pip install torchmetrics albumentations
```

For MixStyle augmentation:
- No additional installation needed - we've implemented MixStyle directly in `src/mixstyle.py`

## Testing the Changes

To test the changes, you can use the following command:

```bash
# Test with all new features enabled
python src/main.py --override use_advanced_aug=True --override use_balanced_sampler=True --override use_projection_head=True --override use_improved_classifier=True --override use_margin_loss=True --override use_learned_mask_token=True --override use_separate_pos_embed=True

# Test with LoRA for parameter-efficient fine-tuning
python src/main.py --override use_lora=True --override freeze_backbone=True --override learning_rate=1e-4
```

## What to Expect

With these changes, you can expect:

1. **Better Performance**: The improved architecture and training techniques should lead to better classification accuracy.

2. **More Efficient Training**: LoRA and other parameter-efficient techniques reduce the number of trainable parameters and make training more efficient.

3. **Better Generalization**: Advanced augmentations and balanced sampling help the model generalize better to unseen data.

4. **More Flexibility**: The modular design and configuration options make it easier to experiment with different approaches.

5. **Better Representation Learning**: The learned mask token, separate positional embeddings, and projection head improve the model's ability to learn meaningful representations.

These changes address the limitations in the original implementation and provide a more robust and flexible framework for morph detection.
