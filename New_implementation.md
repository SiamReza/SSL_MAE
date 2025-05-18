# Implementation Plan for SSL_MAE Improvements

This document outlines a comprehensive plan for improving the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection. Each improvement addresses specific limitations in the current implementation and is designed to enhance performance, efficiency, or generalization.

## 1. Mask-ratio Hard-coded in the Training Loop

**Current Implementation:**
- The mask ratio is defined in `config.py` as `self.mae_mask_ratio = 0.75`
- This value is passed to the ViTMAEConfig when initializing the model
- There's no easy way to experiment with different mask ratios without modifying the code

**Proposed Improvement:**
- Add a CLI flag to allow easy experimentation with different mask ratios
- Modify `config.py` to include a command-line argument for mask ratio
- Update the argument parser in `get_config()` to accept this parameter

**Benefits:**
- Enables quick ablation studies to find the optimal mask ratio for different datasets
- Makes experimentation more accessible without code modifications
- Follows best practices for configuration management

## 2. Mask Generation Code Mixed with Augmentations

**Current Implementation:**
- Masking is handled by the Hugging Face Transformers library's ViTMAEModel
- The masking logic is embedded in the model's forward pass
- This makes it difficult to test or modify the masking strategy independently

**Proposed Improvement:**
- Create a new file `src/mask_utils.py`
- Implement a standalone `random_masking()` function that takes a tensor and mask ratio
- Refactor the model to use this function instead of relying on the built-in masking

**Benefits:**
- Decouples masking logic from the model implementation
- Makes the code more modular, testable, and reusable
- Allows for experimentation with different masking strategies

## 3. No Learned Mask Token in the Decoder

**Current Implementation:**
- There is no decoder implementation or learned mask token
- The current approach doesn't reconstruct masked patches properly
- The reconstruction loss is a simplified version that doesn't use a decoder

**Proposed Improvement:**
- Add a learnable mask token parameter to the model:
  ```python
  self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
  ```
- Implement a proper decoder that uses this mask token
- Replace masked patches with copies of the mask token in the decoder's input

**Benefits:**
- Follows the original MAE paper's approach for better reconstruction
- Improves the model's ability to learn meaningful representations
- Enables proper reconstruction of masked patches

## 4. Single Positional Embedding Used for Both Encoder & Decoder

**Current Implementation:**
- Positional embeddings are handled by the Hugging Face Transformers library
- The same embeddings are used for both encoding and decoding
- This limits the model's flexibility in learning task-specific positional information

**Proposed Improvement:**
- Implement separate positional embeddings for encoder and decoder:
  ```python
  self.pos_embed_enc = nn.Parameter(torch.zeros(1, num_visible+1, D))
  self.pos_embed_dec = nn.Parameter(torch.zeros(1, num_patches+1, D))
  ```
- Use appropriate embeddings in each forward pass
- Update indexing logic to select the right slices for each component

**Benefits:**
- Allows each component to learn optimal positional representations
- Improves both encoding and decoding performance
- Follows the approach used in the original MAE paper

## 5. Missing Projection ("Neck") for Downstream Linear Probe

**Current Implementation:**
- The model directly uses the [CLS] token from the MAE model's last hidden state
- There's no projection head to create a better representation space
- This limits performance on downstream tasks

**Proposed Improvement:**
- Create a new file `src/heads.py` with a ProjectionHead class:
  ```python
  class ProjectionHead(nn.Module):
      def __init__(self, dim, proj_dim=256):
          super().__init__()
          self.net = nn.Sequential(
              nn.LayerNorm(dim),
              nn.Linear(dim, proj_dim),
              nn.GELU(),
              nn.Linear(proj_dim, proj_dim),
          )
      def forward(self, x): return self.net(x)
  ```
- Add this projection head to the model
- Make it optional via a configuration flag

**Benefits:**
- Creates a better representation space for downstream tasks
- Follows the approach of contrastive learning methods like SimCLR
- Improves linear evaluation performance

## 6. Missing Classification Head for Morph vs Bona-fide

**Current Implementation:**
- Uses a simple linear layer for classification:
  ```python
  self.classifier = nn.Linear(self.hidden_size, 1)
  ```
- This may not capture complex relationships in the data
- Limited capacity for the classification task

**Proposed Improvement:**
- Create `src/classifier.py` with a more sophisticated classifier:
  ```python
  class MorphClassifier(nn.Module):
      def __init__(self, embed_dim, num_classes=2):
          super().__init__()
          self.head = nn.Sequential(
              nn.LayerNorm(embed_dim),
              nn.Linear(embed_dim, embed_dim // 2),
              nn.GELU(),
              nn.Linear(embed_dim // 2, num_classes)
          )
      def forward(self, x):
          return self.head(x)
  ```
- Replace the simple linear classifier with this MLP

**Benefits:**
- Increases model capacity for the classification task
- Can capture more complex relationships in the data
- Potentially improves classification performance

## 7. No Margin-Based Loss to Increase Inter-Class Separability

**Current Implementation:**
- Uses binary cross-entropy loss:
  ```python
  cls_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels)
  ```
- No explicit margin between classes
- May not create optimal separation between classes

**Proposed Improvement:**
- Install torchmetrics: `pip install torchmetrics`
- Import MarginLoss: `from torchmetrics.classification import MarginLoss`
- Replace the loss function:
  ```python
  criterion = MarginLoss(num_classes=2, margin=0.3, weight=torch.tensor([1.0,1.0]))
  loss = criterion(logits, labels)
  ```

**Benefits:**
- Enforces a margin between class embeddings
- Increases robustness by pushing classes further apart
- Can improve generalization to unseen data

## 8. Parameter-Efficient Fine-Tuning with LoRA

**Current Implementation:**
- Uses a freezing strategy with gradual unfreezing
- Fine-tunes the entire backbone after initial training
- This can lead to overfitting and inefficient training

**Proposed Improvement:**
- Add a new file `src/lora.py` with LoRA implementation
- Apply LoRA adapters to attention layers:
  ```python
  def apply_lora(model, r=4, alpha=32):
      for name, m in model.named_modules():
          if isinstance(m, nn.MultiheadAttention):
              LoRA(m, r=r, lora_alpha=alpha)
      return model
  ```
- Freeze the backbone and only train LoRA parameters

**Benefits:**
- Reduces the number of trainable parameters
- Prevents overfitting, especially with limited data
- Makes training more efficient

## 9. No Balanced/Diverse Data Loader for Morph vs Bona-fide

**Current Implementation:**
- Simply combines and shuffles bonafide and morph images
- No class weighting or balanced sampling
- May lead to bias towards the majority class

**Proposed Improvement:**
- Implement a WeightedRandomSampler in `data_loader.py`:
  ```python
  def make_balanced_sampler(labels):
      class_counts = np.bincount(labels)
      weights = 1. / class_counts
      sample_weights = weights[labels]
      return WeightedRandomSampler(sample_weights, len(sample_weights))
  ```
- Use this sampler in the DataLoader instead of shuffle=True

**Benefits:**
- Ensures balanced class representation during training
- Prevents bias towards the majority class
- Improves performance on minority classes

## 10. No Advanced Augmentations for Domain Generalization

**Current Implementation:**
- Uses basic augmentations like random rotation and color jitter
- Limited ability to generalize across different domains
- May not be robust to variations in capture devices

**Proposed Improvement:**
- Install required libraries: `pip install timm albumentations`
- Implement advanced augmentations:
  ```python
  import albumentations as A
  from albumentations.pytorch import ToTensorV2
  from mixstyle_pytorch import MixStyle

  train_aug = A.Compose([
      A.RandomResizedCrop(224, 224, scale=(0.8,1.0)),
      A.HorizontalFlip(),
      A.RandAugment(num_ops=2, magnitude=9),
      A.Normalize(),
      ToTensorV2(),
  ])
  ```
- Add MixStyle for domain generalization

**Benefits:**
- Improves generalization across different domains
- Makes the model more robust to variations in capture devices
- Can lead to better performance on unseen data

## Implementation Priority

The improvements are organized by priority and feasibility:

**High Priority and Easy to Implement:**
1. CLI Flag for Mask Ratio
2. Balanced Data Loader
3. Advanced Augmentations
4. Margin-Based Loss

**Medium Priority or Moderate Complexity:**
5. Improved Classification Head
6. Projection Head for Linear Probing
7. Separate Mask Generation Utility

**High Complexity or Requires Significant Changes:**
8. Learned Mask Token in Decoder
9. Separate Positional Embeddings
10. Parameter-Efficient Fine-Tuning with LoRA

This implementation plan addresses the current limitations in the codebase and provides a roadmap for improving the model's performance, efficiency, and generalization capabilities.
