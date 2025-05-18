# SSL_MAE Optimization Changes

This document outlines the changes made to optimize the SSL_MAE codebase for better performance and stability.

## Configuration Parameter Optimization

Based on dataset analysis, the following parameters were optimized in `src/config.py`:

### Class Weights

```python
self.class_weights = [1.37, 0.79]  # Class weights for loss function [bonafide_weight, morph_weight]
# Note: MorDiff dataset has different class distribution with weights [2.73, 0.61]
```

The class weights were set based on the analysis of the dataset distributions:
- Most datasets (LMA, LMA_UBO, MIPGAN_I, MIPGAN_II) have similar class distributions with around 36-37% bonafide images and 63-64% morph images.
- The recommended class weights (average across all datasets except MorDiff) are [1.37, 0.79].
- MorDiff has a different distribution (18.3% bonafide, 81.7% morph) with weights [2.73, 0.61].

### Training Hyperparameters

```python
self.num_epochs = 30  # Increased for better convergence on small datasets
self.batch_size = 16  # Reduced to have more update steps on small datasets
self.learning_rate = 1e-4  # Slightly increased for faster convergence
```

These parameters were optimized based on the dataset sizes:
- The datasets have around 1,800-1,900 images each (except MorDiff which has 3,785).
- A smaller batch size (16) allows for more update steps on these relatively small datasets.
- A moderate learning rate (1e-4) balances convergence speed and stability.
- Increased epochs (30) ensure convergence without overfitting.

### Model Architecture Options

```python
self.use_projection_head = True  # Whether to use projection head for downstream tasks
self.proj_dim = 384  # Increased projection dimension for better representation
self.use_improved_classifier = True  # Enabled improved classifier for better performance
```

These changes improve the model's representation learning and classification capabilities.

### MAE-Specific Options

```python
self.use_learned_mask_token = True  # Enabled learned mask token in decoder for better reconstruction
self.use_separate_pos_embed = True  # Enabled separate positional embeddings for encoder and decoder
```

These options follow the original MAE paper's approach for better reconstruction and representation learning.

## Code Improvements

Several improvements were made to fix potential issues and make the code more robust:

### 1. Fixed Mask Token Reconstruction

The `_reconstruct_from_masked` method in `src/model.py` was improved to properly handle visible token indexing:

```python
for b in range(batch_size):
    visible_idx = 0  # Reset for each batch
    for i in range(num_patches):
        if visible_mask[b, i] == 1:  # This is a visible token
            if visible_idx < visible_tokens.shape[1]:  # Check bounds
                tokens[b, ids_restore[b, i]] = visible_tokens[b, visible_idx]
                visible_idx += 1
        else:  # This is a masked token
            tokens[b, ids_restore[b, i]] = self.mask_token  # Use mask token
```

Key improvements:
- Reset `visible_idx` for each batch to prevent index errors
- Added bounds check to prevent accessing indices beyond available visible tokens
- Fixed mask token usage by removing unnecessary squeeze operation

### 2. Improved Margin Loss Implementation

Added a fallback implementation for margin loss when torchmetrics is not available:

```python
# Fallback implementation of MarginLoss
class MarginLoss(nn.Module):
    def __init__(self, num_classes=2, margin=0.3, weight=None):
        super().__init__()
        self.margin = margin
        self.weight = weight
        
    def forward(self, logits, target):
        # Simple margin loss implementation
        # For binary classification: max(0, margin - y * f(x))
        if logits.shape[-1] == 2:  # Two-class logits
            pos_logits = logits[:, 1]
            neg_logits = logits[:, 0]
        else:  # Single output
            pos_logits = logits.squeeze()
            neg_logits = -logits.squeeze()
            
        # Convert target to -1/1
        target_pm = 2 * target.float() - 1
        
        # Calculate loss
        loss = torch.mean(torch.clamp(self.margin - target_pm * (pos_logits - neg_logits), min=0))
        
        return loss
```

### 3. Enhanced Binary Cross-Entropy Loss

Improved the binary cross-entropy loss to properly use class weights:

```python
# Use standard binary cross-entropy loss with class weights
if hasattr(self.config, 'class_weights') and len(self.config.class_weights) == 2:
    # Apply class weights to BCE loss
    pos_weight = torch.tensor([self.config.class_weights[1] / self.config.class_weights[0]], 
                             device=logits.device)
    cls_loss = F.binary_cross_entropy_with_logits(
        logits.squeeze(), labels, pos_weight=pos_weight
    )
else:
    # Standard BCE loss without weights
    cls_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels)
```

## Conclusion

These optimizations and improvements should result in:
1. Better handling of class imbalance through appropriate class weights
2. Improved training efficiency with optimized hyperparameters
3. Enhanced model architecture with projection head and improved classifier
4. More robust code with fixed potential issues
5. Better reconstruction with learned mask token and separate positional embeddings

The changes maintain compatibility with the existing codebase while improving performance and stability.
