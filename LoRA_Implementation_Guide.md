# LoRA Implementation Guide for SSL_MAE

This document provides a guide for using Low-Rank Adaptation (LoRA) in the SSL_MAE codebase to prevent overfitting and improve model performance.

## What is LoRA?

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that significantly reduces the number of trainable parameters while maintaining performance. Instead of fine-tuning all weights in a pre-trained model, LoRA:

1. Freezes the original model weights
2. Injects trainable rank decomposition matrices into each layer
3. Updates only these small adapter matrices during training

This approach is particularly useful for preventing overfitting when fine-tuning large models on small datasets.

## How LoRA is Implemented in SSL_MAE

In the SSL_MAE codebase, LoRA is implemented using the PEFT (Parameter-Efficient Fine-Tuning) library from Hugging Face. The implementation:

1. Freezes the backbone ViT MAE model
2. Adds LoRA adapters to the attention modules (query, key, value projections)
3. Only trains these adapters and the classification head

## Configuration Parameters

The following parameters in `config.py` control LoRA behavior:

```python
# LoRA options
self.use_lora = True  # Whether to use LoRA for parameter-efficient fine-tuning
self.lora_r = 4  # Rank for LoRA adaptation
self.lora_alpha = 32  # Alpha parameter for LoRA
self.lora_dropout = 0.1  # Dropout probability for LoRA layers
self.lora_target_modules = ["query", "key", "value"]  # Target modules for LoRA adaptation
```

### Parameter Explanation

- **use_lora**: Enables or disables LoRA
- **lora_r**: The rank of the low-rank decomposition. Lower values mean fewer parameters but potentially less expressive power
- **lora_alpha**: Scaling factor for the LoRA updates. Higher values give more weight to the LoRA updates
- **lora_dropout**: Dropout probability applied to the LoRA layers for regularization
- **lora_target_modules**: Which modules to apply LoRA to. In ViT models, these are typically the attention modules

## Recommended Settings Based on Dataset Size

| Dataset Size | Recommended lora_r | Recommended lora_alpha |
|--------------|-------------------|------------------------|
| Small (<1000 images) | 4 | 32 |
| Medium (1000-5000 images) | 8 | 32 |
| Large (>5000 images) | 16 | 32 |

## How to Use LoRA

### Enabling LoRA

LoRA is now enabled by default in the updated configuration. If you want to explicitly enable it:

```bash
python src/train.py --override use_lora=True
```

### Disabling LoRA

If you want to disable LoRA and use traditional fine-tuning:

```bash
python src/train.py --override use_lora=False
```

### Adjusting LoRA Parameters

To adjust the rank of LoRA:

```bash
python src/train.py --override lora_r=8
```

To adjust multiple parameters:

```bash
python src/train.py --override lora_r=8 --override lora_alpha=16
```

## Benefits of Using LoRA

1. **Prevents Overfitting**: By reducing the number of trainable parameters, LoRA helps prevent overfitting, especially on small datasets
2. **Faster Training**: Training fewer parameters leads to faster training times
3. **Lower Memory Usage**: Requires less GPU memory during training
4. **Better Generalization**: Often leads to better generalization to unseen data
5. **Preserves Pre-trained Knowledge**: Keeps the valuable knowledge in the pre-trained model while adapting to the new task

## When to Use LoRA

LoRA is particularly useful when:

1. Your dataset is small compared to the model size
2. You're fine-tuning a large pre-trained model
3. You're experiencing overfitting with traditional fine-tuning
4. You have limited computational resources

## Monitoring LoRA Performance

When using LoRA, monitor:

1. The validation loss and accuracy to ensure the model is learning effectively
2. The gap between training and validation metrics to detect overfitting
3. The final performance on test data to evaluate generalization

If LoRA performance is not satisfactory, consider:

1. Adjusting the rank (lora_r)
2. Changing which modules LoRA is applied to
3. Combining LoRA with other regularization techniques like dropout and weight decay

## Conclusion

LoRA is a powerful technique for fine-tuning large models on small datasets while preventing overfitting. The implementation in SSL_MAE makes it easy to use and configure for optimal performance on morph detection tasks.
