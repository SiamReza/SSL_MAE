# LoRA Implementation Guide

This document provides detailed information about the Low-Rank Adaptation (LoRA) implementation in the SSL_MAE codebase, including how to use it effectively for morph detection tasks.

## What is LoRA?

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that significantly reduces the number of trainable parameters while maintaining performance. Instead of fine-tuning all weights in a pre-trained model, LoRA:

1. Freezes the original model weights
2. Injects trainable rank decomposition matrices into each layer
3. Updates only these small adapter matrices during training

This approach has several advantages:
- Reduces trainable parameters by >99% in most cases
- Prevents overfitting on small datasets
- Requires less memory and computation
- Enables faster training and inference
- Results in much smaller checkpoint files

## How LoRA Works

For a pre-trained weight matrix W ∈ ℝᵐˣⁿ, LoRA parameterizes the update with:

ΔW = BA

Where:
- B ∈ ℝᵐˣʳ and A ∈ ℝʳˣⁿ
- r is the rank (typically r << min(m,n))

During the forward pass, the computation becomes:

h = Wx + ΔWx = Wx + BAx

The original weights W remain frozen, while only A and B are trained. The rank r controls the capacity of the adaptation.

## LoRA Parameters in SSL_MAE

The SSL_MAE codebase implements LoRA with the following configurable parameters:

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `use_lora` | Whether to use LoRA | `True` | `True` for parameter-efficient fine-tuning |
| `lora_r` | Rank for LoRA adaptation | `4` | `4` for smaller datasets, `8` for larger datasets |
| `lora_alpha` | Scaling factor for LoRA | `32` | `32` (works well with rank 4) |
| `lora_dropout` | Dropout probability for LoRA layers | `0.1` | `0.1` for smaller datasets, `0.05` for larger datasets |
| `lora_target_modules` | Target modules for LoRA adaptation | `["query", "key", "value"]` | Same as default |

## How to Train with LoRA

### Basic Training Command

```bash
python src/train.py
```

LoRA is enabled by default in the configuration. To explicitly enable it:

```bash
python src/train.py --override use_lora=True --override freeze_strategy=backbone_only
```

### Customizing LoRA Parameters

You can customize the LoRA parameters using the `--override` flag:

```bash
python src/train.py --override lora_r=8 --override lora_alpha=32 --override lora_dropout=0.05
```

### Recommended Settings for Different Dataset Sizes

#### For Small Datasets (<2000 images)
```bash
python src/train.py --override lora_r=4 --override lora_alpha=32 --override lora_dropout=0.1 --override learning_rate=5e-4
```

#### For Medium Datasets (2000-5000 images)
```bash
python src/train.py --override lora_r=8 --override lora_alpha=32 --override lora_dropout=0.05 --override learning_rate=5e-4
```

#### For Large Datasets (>5000 images)
```bash
python src/train.py --override lora_r=16 --override lora_alpha=32 --override lora_dropout=0.05 --override learning_rate=3e-4
```

## How to Test LoRA Models

When testing a model trained with LoRA, you must ensure that LoRA is enabled during testing:

```bash
python src/test.py --model_path output/models/LMA_morphdetector.pt --override use_lora=True
```

You should also use the same LoRA parameters that were used during training:

```bash
python src/test.py --model_path output/models/LMA_morphdetector.pt --override use_lora=True --override lora_r=8
```

## Comparing LoRA vs. Full Fine-tuning

### Advantages of LoRA

1. **Parameter Efficiency**: LoRA typically reduces trainable parameters by >99%
2. **Memory Efficiency**: Requires significantly less GPU memory
3. **Training Speed**: Faster training due to fewer parameter updates
4. **Overfitting Prevention**: Less prone to overfitting on small datasets
5. **Storage Efficiency**: Smaller checkpoint files

### When to Use LoRA

LoRA is particularly beneficial in the following scenarios:

1. **Small Datasets**: When you have limited training data (e.g., <5000 images)
2. **Limited Compute Resources**: When you have limited GPU memory or compute power
3. **Quick Adaptation**: When you need to quickly adapt a model to a new domain
4. **Multiple Adaptations**: When you want to maintain multiple specialized versions of a model

### When to Use Full Fine-tuning

Full fine-tuning might be preferable in these cases:

1. **Large Datasets**: When you have abundant training data (e.g., >10,000 images)
2. **Significant Domain Shift**: When the target domain is very different from the pre-training domain
3. **Maximum Performance**: When you need to extract every bit of performance and have sufficient resources

## Implementation Details

In the SSL_MAE codebase, LoRA is implemented in `src/lora.py` and applied to the attention layers of the Vision Transformer. The implementation:

1. Identifies attention modules in the model
2. Injects LoRA adapters into the query, key, and value projections
3. Freezes the original weights
4. Only updates the LoRA adapter weights during training

The LoRA adapters are initialized with a normal distribution with mean 0 and standard deviation 0.02, following the standard practice.

## Tips for Optimal LoRA Performance

1. **Learning Rate**: Use a higher learning rate with LoRA (typically 5e-4 instead of 1e-4)
2. **Rank Selection**: Start with rank 4 for small datasets, increase to 8 or 16 for larger datasets
3. **Alpha Scaling**: Keep alpha at 32 for rank 4 (effective rank of 8), scale proportionally for other ranks
4. **Target Modules**: For Vision Transformers, targeting query, key, and value projections is most effective
5. **Freezing Strategy**: Use `freeze_strategy=backbone_only` with LoRA
6. **Augmentations**: When using LoRA with a frozen backbone, you can disable MAE-specific augmentations (`use_mae_aug=False`)
7. **Epochs**: LoRA often converges faster, so you might need fewer epochs (20-25 instead of 30)

## Troubleshooting

### Common Issues

1. **Low Performance**: Try increasing the rank (lora_r) or using a higher learning rate
2. **Overfitting**: Try increasing lora_dropout or reducing the rank
3. **Slow Convergence**: Ensure you're using a higher learning rate with LoRA (5e-4 recommended)
4. **Memory Errors**: Ensure the backbone is properly frozen (`freeze_strategy=backbone_only`)

### Verifying LoRA is Working

To verify that LoRA is working correctly:

1. Check the model summary to confirm the number of trainable parameters is small
2. Monitor training loss to ensure it's decreasing
3. Verify that only LoRA adapter weights are being updated (the backbone weights should remain unchanged)

## Conclusion

LoRA is a powerful technique for parameter-efficient fine-tuning that works particularly well for morph detection tasks with limited data. By following the guidelines in this document, you can effectively use LoRA to achieve good performance while using fewer computational resources.
