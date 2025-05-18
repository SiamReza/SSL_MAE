# SSL_MAE: Self-Supervised Vision-Transformer for Morph-Attack Detection

This repository contains a self-supervised learning pipeline based on Masked Autoencoders (MAE) for morph attack detection. The codebase supports both standard fine-tuning and parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA).

## Table of Contents

- [Setup](#setup)
- [Dataset Structure](#dataset-structure)
- [Training](#training)
  - [Standard Training](#standard-training)
  - [Training with LoRA](#training-with-lora)
- [Testing](#testing)
  - [Standard Testing](#standard-testing)
  - [Testing LoRA Models](#testing-lora-models)
- [Configuration](#configuration)
- [Tips for Optimal Performance](#tips-for-optimal-performance)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SSL_MAE.git
cd SSL_MAE
```

2. Install dependencies:
```bash
pip install torch torchvision tqdm matplotlib numpy pandas scikit-learn
pip install transformers albumentations
```

## Dataset Structure

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

Where `{dataset_name}` is one of the supported datasets (e.g., "LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN_IWBF").

## Training

### Standard Training

To train the model with standard fine-tuning (without LoRA):

```bash
python src/train.py --override use_lora=False
```

This will train the model using the default configuration in `src/config.py`. You can override any configuration parameter using the `--override` flag:

```bash
python src/train.py --override use_lora=False --override learning_rate=1e-4 --override num_epochs=30
```

### Training with LoRA

To train the model with parameter-efficient fine-tuning using LoRA:

```bash
python src/train.py --override use_lora=True --override freeze_strategy=backbone_only
```

LoRA is enabled by default in the configuration, but you can customize the LoRA parameters:

```bash
python src/train.py --override lora_r=8 --override lora_alpha=32 --override learning_rate=5e-4
```

#### Key LoRA Parameters

- `lora_r`: Rank for LoRA adaptation (default: 4)
- `lora_alpha`: Scaling factor (default: 32)
- `lora_dropout`: Dropout probability (default: 0.1)
- `learning_rate`: Typically higher for LoRA (5e-4 recommended)

#### Differences from Standard Training

When using LoRA:
1. The backbone model remains frozen, and only the LoRA adapters and classification head are updated
2. Training is faster and requires less memory
3. The model is less prone to overfitting, especially on smaller datasets
4. Saved checkpoints are much smaller (only storing LoRA adapter weights)
5. Higher learning rates can be used (typically 5e-4 instead of 1e-4)

## Testing

### Standard Testing

To test a trained model on all datasets:

```bash
python src/test.py --model_path output/models/LMA_morphdetector.pt
```

To test on specific datasets:

```bash
python src/test.py --model_path output/models/LMA_morphdetector.pt --datasets LMA MIPGAN_I
```

### Testing LoRA Models

To test a model trained with LoRA, make sure to enable LoRA during testing:

```bash
python src/test.py --model_path output/models/LMA_morphdetector.pt --override use_lora=True
```

You can also specify a custom threshold for binary classification:

```bash
python src/test.py --model_path output/models/LMA_morphdetector.pt --override use_lora=True --threshold 0.7
```

## Configuration

All configuration parameters are defined in `src/config.py`. See `Config_desc.md` for detailed descriptions of each parameter.

Key configuration parameters:

```python
# Data parameters
self.batch_size = 16  # Batch size for training
self.train_dataset = "LMA"  # Dataset for training

# Model architecture
self.use_projection_head = True  # Whether to use projection head
self.use_improved_classifier = True  # Whether to use improved classifier

# Training parameters
self.num_epochs = 30  # Number of training epochs
self.learning_rate = 1e-4  # Learning rate (5e-4 for LoRA)

# Loss function
self.use_margin_loss = True  # Whether to use margin-based loss
self.class_weights = [1.37, 0.79]  # Class weights [bonafide, morph]

# LoRA parameters
self.use_lora = True  # Whether to use LoRA
self.lora_r = 4  # Rank for LoRA adaptation
self.lora_alpha = 32  # Alpha parameter for LoRA
```

## Tips for Optimal Performance

1. **Dataset Size Considerations**:
   - For smaller datasets (<2000 images), use LoRA with `lora_r=4`
   - For larger datasets (>2000 images), you can use standard fine-tuning or LoRA with `lora_r=8`

2. **Class Imbalance**:
   - Enable `use_balanced_sampler=True` for imbalanced datasets
   - Use appropriate `class_weights` based on your dataset distribution
   - For MorDiff dataset, use class weights `[2.73, 0.61]`

3. **Learning Rate**:
   - For standard fine-tuning: `learning_rate=1e-4`
   - For LoRA: `learning_rate=5e-4`

4. **Augmentations**:
   - Enable `use_advanced_aug=True` for better generalization
   - When using LoRA with frozen backbone, set `use_mae_aug=False`

5. **Model Architecture**:
   - Enable `use_projection_head=True` and `use_improved_classifier=True` for better performance
   - For MAE features, enable `use_learned_mask_token=True` and `use_separate_pos_embed=True`

6. **Evaluation**:
   - Test on multiple datasets to assess generalization
   - Pay attention to metrics like AUC, EER, and BPCER at different APCER thresholds

For more detailed information about each configuration parameter, refer to `Config_desc.md`.
