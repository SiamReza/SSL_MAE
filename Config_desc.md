# SSL_MAE Configuration Parameters Guide

This document provides comprehensive descriptions of all configuration parameters in the SSL_MAE codebase, along with recommended values for optimal performance. Understanding these parameters is crucial for achieving the best results on your specific datasets.

## Introduction

The SSL_MAE codebase implements a self-supervised learning approach based on Masked Autoencoders (MAE) for morph attack detection. The configuration system is designed to be flexible and allow for easy experimentation with different settings.

### Key Concepts

- **Self-Supervised Learning (SSL)**: The model learns useful representations from the data itself without explicit labels by reconstructing masked portions of the input images.
- **Masked Autoencoder (MAE)**: A specific SSL approach where random patches of the input image are masked, and the model learns to reconstruct them.
- **Parameter-Efficient Fine-Tuning**: Techniques like LoRA that allow adapting pre-trained models with minimal trainable parameters.
- **Morph Attack Detection**: The task of identifying morphed facial images (artificially combined faces) from bonafide (genuine) images.

### Dataset Structure

The codebase expects datasets to be organized in the following structure:

```
datasets/
├── bonafide/
│   ├── {dataset_name}/
│   │   ├── train/
│   │   │   └── *.jpg
│   │   └── test/
│   │       └── *.jpg
└── morph/
    ├── {dataset_name}/
    │   ├── train/
    │   │   └── *.jpg
    │   └── test/
    │       └── *.jpg
```

Where `{dataset_name}` is one of the supported datasets:
- LMA
- LMA_UBO
- MIPGAN_I
- MIPGAN_II
- MorDiff
- StyleGAN

### How to Use This Guide

This guide is organized into sections based on parameter categories. For each parameter, you'll find:
- A detailed description of what it does
- The default value
- Recommended values for different scenarios
- How it interacts with other parameters
- Examples of when and how to adjust it

Use the table of contents to navigate to the specific parameters you're interested in.

## Basic Configuration Parameters

### Data Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `data_root` | Root directory containing the dataset folders. This should point to the parent directory that contains the "bonafide" and "morph" subdirectories. | `"datasets"` | Path to your dataset root directory |
| `train_dataset` | Name of the dataset to use for training. This corresponds to a subdirectory name within the bonafide/morph folders (e.g., "LMA", "MIPGAN_I"). | `"LMA"` | Choose based on your available data and task requirements |
| `test_datasets` | List of dataset names to use for testing. The model will be evaluated on each of these datasets separately. These names must match the directory names in your dataset structure. | `["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN"]` | Include all datasets you want to evaluate on. For cross-dataset evaluation, include datasets not used in training. |
| `train_val_pct` | Percentage of training data to use for training (vs. validation). For example, 0.8 means 80% of the data will be used for training and 20% for validation. | `0.8` | `0.8` - `0.9` (higher values for smaller datasets) |
| `batch_size` | Number of samples processed in each training/evaluation batch. Larger batch sizes can lead to faster training but require more GPU memory. | `16` | `16` - `64` (adjust based on GPU memory; use smaller values for larger models) |

### Model Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `pretrained_model` | Specifies which pre-trained model to use as the backbone. Options are "vit_mae" (Meta AI's MAE model) or "webssl" (Google's WebSSL model). | `"vit_mae"` | `"vit_mae"` for general use, `"webssl"` for more robust features |
| `dropout_p` | Probability of dropping units during training to prevent overfitting. Higher values (e.g., 0.5) provide more regularization but may slow down convergence. | `0.5` | `0.3` - `0.5` (higher for smaller datasets, lower for larger ones) |
| `freeze_strategy` | Strategy for freezing/unfreezing backbone layers during training. Options: "none" (train all), "backbone_only" (freeze backbone), "freeze_except_lastN" (freeze all except last N blocks), "gradual_unfreeze" (gradually unfreeze layers). | `"gradual_unfreeze"` | `"gradual_unfreeze"` for full fine-tuning, `"backbone_only"` for LoRA |
| `warmup_epochs` | Number of initial epochs to train only the classification head before unfreezing other layers (used with "gradual_unfreeze"). | `2` | `2` - `5` (higher for more complex datasets) |
| `mid_epochs` | Epoch at which to start unfreezing the full backbone (used with "gradual_unfreeze"). | `5` | `5` - `10` (adjust based on total epochs) |
| `freeze_lastN` | Number of final transformer blocks to keep trainable when using "freeze_except_lastN" strategy. | `2` | `1` - `4` (higher values allow more adaptation) |

### Training Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `num_epochs` | Total number of training epochs (complete passes through the training dataset). More epochs generally lead to better performance, but too many can cause overfitting. | `30` | `20` - `50` (adjust based on dataset size; use early stopping if available) |
| `learning_rate` | Step size for gradient descent during optimization. Controls how quickly the model parameters are updated. Too high can cause divergence, too low can lead to slow convergence. | `1e-4` | `1e-4` for full fine-tuning, `5e-4` for LoRA (higher learning rates are typically better for LoRA since fewer parameters are being updated) |
| `recon_weight` | Weight for the reconstruction loss component in the total loss function. Higher values emphasize better reconstruction of masked patches, which can improve representation learning. | `0.2` | `0.1` - `0.5` (higher for better self-supervised learning, lower for more focus on classification) |
| `lr_scheduler` | Type of learning rate scheduler to use. Options: "cosine" (gradually decreases LR following a cosine curve) or "none" (constant LR). | `"cosine"` | `"cosine"` (helps convergence and often leads to better final performance) |
| `eta_min` | Minimum learning rate at the end of cosine schedule. The learning rate will decay from `learning_rate` to `eta_min` over the course of training. | `0.0` | `0.0` - `1e-6` (small non-zero value can help prevent stagnation) |
| `T_max` | Number of epochs over which to decay the learning rate when using cosine scheduler. | `num_epochs` | Same as `num_epochs` (full training duration) |
| `use_fixed_seed` | Whether to use a fixed seed for reproducibility. Setting to True ensures the same results across runs, while False introduces randomness. | `True` | `True` for development/debugging, `False` for final training or ensemble models |
| `seed` | Seed value when `use_fixed_seed` is True. Controls the initialization of random number generators for reproducibility. | `42` | Any integer (42 is traditional, but any value works) |

## Advanced Configuration Parameters

### Augmentation Options

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `use_cls_aug` | Whether to use basic augmentations (random crop, flip, etc.) for the classifier. These augmentations help improve generalization by exposing the model to variations of the input images. | `True` | `True` (almost always beneficial) |
| `use_mae_aug` | Whether to use MAE-specific augmentations, which include masking random patches of the input image during training. This is a key component of the MAE self-supervised learning approach. | `True` | `True` for self-supervised learning, `False` when using LoRA with frozen backbone |
| `mae_mask_ratio` | Percentage of image patches to mask during MAE training. Higher values make the task more difficult but can lead to better representations. The original MAE paper used 0.75. | `0.75` | `0.6` - `0.8` (0.75 is a good balance; higher values for more challenging pretraining) |
| `use_advanced_aug` | Whether to use advanced augmentations from the albumentations library, including color jitter, blur, and distortion. These help improve robustness to variations in image quality and capture conditions. | `True` | `True` for better generalization to unseen data and different domains |
| `advanced_aug_magnitude` | Strength of the RandAugment augmentations (1-10). Higher values apply stronger transformations, which can improve generalization but may make training more difficult. | `9` | `7` - `10` (higher for more aggressive augmentation, lower for preserving more original content) |
| `advanced_aug_num_ops` | Number of augmentation operations to apply in sequence with RandAugment. More operations increase diversity but may distort images too much. | `2` | `2` - `3` (2 is usually sufficient; 3 for more aggressive augmentation) |

### Data Loading Options

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `use_balanced_sampler` | Whether to use a weighted random sampler that balances class frequencies during training. This ensures that each class is equally represented in training batches, which is important for imbalanced datasets where one class (e.g., morph or bonafide) appears more frequently than the other. The sampler assigns weights inversely proportional to class frequencies. | `True` | `True` for imbalanced datasets (almost always beneficial for morph detection where class distributions are often skewed) |

### Model Architecture Options

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `use_projection_head` | Whether to use a projection head between the encoder and classifier. The projection head is a small MLP that transforms the encoder's output into a representation space better suited for the downstream task. It follows the approach used in contrastive learning methods like SimCLR. | `True` | `True` for better representations and downstream performance (especially important for transfer learning) |
| `proj_dim` | Output dimension of the projection head. This determines the size of the representation space. Larger dimensions can capture more information but increase parameter count. | `384` | `256` - `512` (384 is a good balance; larger values for more complex datasets) |
| `proj_hidden_dim` | Hidden dimension for the projection head's intermediate layer. If None, it defaults to half of the input dimension. | `None` | `None` (the default automatic sizing works well in most cases) |
| `use_improved_classifier` | Whether to use an improved classifier with a hidden layer instead of a simple linear layer. The improved classifier has better capacity to learn complex decision boundaries. | `True` | `True` for better classification performance (especially for challenging datasets) |
| `cls_hidden_dim` | Hidden dimension for the improved classifier. If None, it defaults to half of the input dimension. | `None` | `None` (the default automatic sizing works well in most cases) |

### Loss Function Options

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `use_margin_loss` | Whether to use margin-based loss instead of standard binary cross-entropy. Margin loss enforces a minimum margin between class embeddings, which can improve separation between bonafide and morph samples. It's particularly useful for challenging datasets where the classes are difficult to separate. | `True` | `True` for better class separation and generalization to unseen data |
| `margin` | The minimum margin to enforce between positive and negative samples in the margin loss. Larger margins enforce stronger separation but may make optimization more difficult. | `0.3` | `0.3` - `0.5` (0.3 is a good starting point; increase for more challenging datasets) |
| `class_weights` | Weights applied to each class in the loss function to handle class imbalance. Format is [bonafide_weight, morph_weight]. Higher weight for a class means the model will prioritize correctly classifying that class. Based on dataset analysis, the optimal weights are [1.37, 0.79] for most datasets, with MorDiff having different weights [2.73, 0.61] due to its more extreme class imbalance. | `[1.37, 0.79]` | `[1.37, 0.79]` for most datasets; adjust based on your specific class distribution |

### MAE-Specific Options

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `use_learned_mask_token` | Whether to use a learnable mask token in the decoder for reconstructing masked patches. This follows the original MAE paper's approach where a special learned token is used to represent masked patches during reconstruction. When enabled, the model learns a more meaningful representation for masked content. | `True` | `True` for better reconstruction quality and representation learning |
| `use_separate_pos_embed` | Whether to use separate positional embeddings for the encoder and decoder. This allows each component to learn optimal positional representations for their specific tasks. The encoder needs positional information for understanding image structure, while the decoder needs it for accurate reconstruction. | `True` | `True` for better performance in both encoding and reconstruction |
| `decoder_depth` | Number of transformer layers in the decoder. More layers provide greater capacity for reconstruction but increase computational cost. The original MAE paper used a lighter decoder than encoder. | `4` | `4` - `6` (4 is sufficient for most cases; increase for more complex datasets) |

### LoRA Options

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `use_lora` | Whether to use Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning. LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer, drastically reducing the number of trainable parameters (often by >99%) while maintaining performance. This is particularly useful for fine-tuning large models on small datasets to prevent overfitting. | `True` | `True` for parameter-efficient fine-tuning (especially beneficial for smaller datasets) |
| `lora_r` | Rank for LoRA adaptation matrices. This determines the rank of the low-rank decomposition used in LoRA. Lower ranks mean fewer parameters but less expressive power. The rank effectively controls the capacity of the adaptation. For your dataset sizes (1,800-3,800 images), a rank of 4-8 is appropriate. | `4` | `4` for smaller datasets (<2000 images), `8` for larger datasets (>2000 images) |
| `lora_alpha` | Scaling factor for LoRA. This controls the magnitude of the LoRA update. The effective rank is determined by `lora_alpha/lora_r`. Higher values lead to larger updates from the LoRA matrices. | `32` | `32` (works well with rank 4 to give an effective rank of 8) |
| `lora_dropout` | Dropout probability applied in LoRA layers to prevent overfitting. Higher values provide stronger regularization but may slow down convergence. | `0.1` | `0.1` for smaller datasets, `0.05` for larger datasets |
| `lora_target_modules` | List of module types to apply LoRA to. For Vision Transformers, targeting the attention components ("query", "key", "value") is most effective as these contain most of the model's capacity for adaptation. | `["query", "key", "value"]` | `["query", "key", "value"]` (standard approach for Vision Transformers) |

## Recommended Configuration Sets

### Configuration for Best Performance

For the best overall performance, we recommend the following configuration:

```python
# Data parameters
batch_size = 32  # Adjust based on GPU memory

# Augmentation options
use_cls_aug = True
use_mae_aug = True
mae_mask_ratio = 0.75
use_advanced_aug = True
advanced_aug_magnitude = 9
advanced_aug_num_ops = 2

# Data loading options
use_balanced_sampler = True

# Model architecture options
use_projection_head = True
proj_dim = 384
use_improved_classifier = True

# Loss function options
use_margin_loss = True
margin = 0.3
class_weights = [1.0, 1.0]  # Adjust based on class distribution

# MAE-specific options
use_learned_mask_token = True
use_separate_pos_embed = True
decoder_depth = 4

# Training parameters
learning_rate = 1e-4
weight_decay = 0.05
num_epochs = 30
warmup_steps = 1000
recon_weight = 0.2
```

### Configuration for Parameter-Efficient Fine-Tuning (LoRA)

For parameter-efficient fine-tuning with LoRA, we recommend:

```python
# Data parameters
batch_size = 64  # Can use larger batches due to lower memory requirements

# Augmentation options
use_cls_aug = True
use_mae_aug = False  # Not needed with frozen backbone
use_advanced_aug = True
advanced_aug_magnitude = 9
advanced_aug_num_ops = 2

# Data loading options
use_balanced_sampler = True

# Model architecture options
use_projection_head = True
proj_dim = 256
use_improved_classifier = True

# Loss function options
use_margin_loss = True
margin = 0.3

# LoRA options
use_lora = True
freeze_backbone = True
lora_r = 8
lora_alpha = 32
lora_dropout = 0.1

# Training parameters
learning_rate = 5e-4  # Higher learning rate for LoRA
weight_decay = 0.01
num_epochs = 20  # Fewer epochs needed for LoRA
warmup_steps = 500
recon_weight = 0.0  # No reconstruction with frozen backbone
```

### Configuration for Low-Resource Environments

For training with limited computational resources:

```python
# Data parameters
batch_size = 16  # Smaller batch size for less memory

# Augmentation options
use_cls_aug = True
use_mae_aug = False
use_advanced_aug = False  # Simpler augmentations

# Data loading options
use_balanced_sampler = True

# Model architecture options
use_projection_head = False  # Simpler model
use_improved_classifier = True

# Loss function options
use_margin_loss = False  # Simpler loss function

# LoRA options
use_lora = True
freeze_backbone = True
lora_r = 4  # Smaller rank for fewer parameters
lora_alpha = 16

# Training parameters
learning_rate = 1e-3
weight_decay = 0.01
num_epochs = 15
warmup_steps = 200
```

## How to Apply These Configurations

There are several ways to apply and modify configurations in the SSL_MAE codebase:

### Command-Line Overrides

The most flexible approach is to use the `--override` flag when running training or testing scripts:

```bash
python src/train.py --override use_projection_head=True --override use_improved_classifier=True
```

You can override multiple parameters in a single command:

```bash
python src/train.py --override learning_rate=5e-4 --override batch_size=64 --override use_lora=True
```

### Modifying the Config Class

For permanent changes, you can modify the default values in the `Config` class in `src/config.py`:

```python
# In src/config.py
def __init__(self):
    # ...
    self.learning_rate = 5e-4  # Changed from default 1e-4
    self.batch_size = 64       # Changed from default 32
    # ...
```

### Configuration Files (Advanced)

For complex configurations, you can create configuration files and load them:

1. Create a JSON configuration file:
```json
{
  "learning_rate": 5e-4,
  "batch_size": 64,
  "use_lora": true,
  "lora_r": 8
}
```

2. Load it using the `--config_file` argument (requires implementing a config file loader):
```bash
python src/train.py --config_file my_config.json
```

### Parameter Type Handling

The configuration system automatically handles different parameter types:
- **Boolean values**: Use `true` or `false` (case-insensitive)
- **Numeric values**: Integers and floats are automatically converted
- **Lists**: Use comma-separated values (e.g., `--override test_datasets=LMA,MIPGAN_I`)
- **Strings**: No special handling needed

### Checking Current Configuration

To check the current configuration without running training:

```bash
python -c "from src.config import get_config; print(get_config())"
```

This will print all configuration parameters with their current values.

## Hyperparameter Tuning

For optimal results on your specific dataset, we recommend systematic tuning of key hyperparameters. This section provides guidance on which parameters to prioritize and how to approach the tuning process.

### Priority Parameters for Tuning

1. **`learning_rate`**: The most important parameter to tune
   - Range: 1e-5 to 1e-3
   - Standard training: Start with 1e-4 and adjust up/down by factors of 3
   - LoRA training: Start with 5e-4 and adjust up/down by factors of 2
   - Impact: Directly affects convergence speed and final performance
   - Example: `python src/train.py --override learning_rate=3e-4`

2. **`mae_mask_ratio`**: Controls the difficulty of the self-supervised task
   - Range: 0.6 to 0.8
   - Lower values (0.6): Easier reconstruction task, faster convergence
   - Higher values (0.8): More challenging task, potentially better representations
   - Impact: Affects the quality of learned representations
   - Example: `python src/train.py --override mae_mask_ratio=0.7`

3. **`lora_r`** (when using LoRA): Controls the capacity of the adaptation
   - Range: 4 to 16
   - Smaller datasets (<2000 images): Use rank 4
   - Medium datasets (2000-5000 images): Use rank 8
   - Larger datasets (>5000 images): Use rank 16
   - Impact: Higher ranks increase capacity but require more data to avoid overfitting
   - Example: `python src/train.py --override use_lora=True --override lora_r=8`

4. **`margin`** (when using margin loss): Controls class separation
   - Range: 0.1 to 0.5
   - Easier datasets: Lower values (0.1-0.3)
   - More challenging datasets: Higher values (0.3-0.5)
   - Impact: Affects how strongly the model separates classes
   - Example: `python src/train.py --override use_margin_loss=True --override margin=0.4`

5. **`recon_weight`**: Balances reconstruction vs. classification
   - Range: 0.1 to 0.5
   - Lower values: More focus on classification
   - Higher values: More focus on representation learning
   - Impact: Affects the balance between the two learning objectives
   - Example: `python src/train.py --override recon_weight=0.3`

### Tuning Methodology

For systematic hyperparameter tuning, we recommend:

1. **Sequential Tuning**: Start with learning rate, then tune other parameters in order of importance
2. **Grid Search**: For 2-3 parameters, try all combinations of a few values for each
3. **Random Search**: For more parameters, randomly sample combinations (often more efficient than grid search)
4. **Cross-Validation**: Use k-fold cross-validation for more reliable results, especially with smaller datasets

Example grid search for learning rate and mask ratio:
```bash
# Try different combinations of learning rate and mask ratio
for lr in 1e-5 3e-5 1e-4 3e-4; do
  for mask in 0.6 0.7 0.8; do
    python src/train.py --override learning_rate=$lr --override mae_mask_ratio=$mask
  done
done
```

### Parameter Interactions

Be aware of important parameter interactions:
- `learning_rate` and `batch_size`: Larger batch sizes often work better with higher learning rates
- `use_lora` and `learning_rate`: LoRA typically requires higher learning rates
- `mae_mask_ratio` and `recon_weight`: Higher mask ratios may need higher reconstruction weights
- `lora_r` and `lora_dropout`: Higher ranks may benefit from higher dropout to prevent overfitting

### Evaluation Metrics for Tuning

When tuning, focus on these metrics (in order of importance):
1. Validation accuracy
2. AUC (Area Under ROC Curve)
3. EER (Equal Error Rate)
4. BPCER at different APCER thresholds (for operational requirements)

The best hyperparameters often vary by dataset, so it's worth investing time in proper tuning for your specific use case.
