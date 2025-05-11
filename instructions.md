# Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection

This document provides a detailed, step-by-step plan for implementing a complete pipeline for morph-attack detection using self-supervised Vision Transformers. The implementation will be divided into four major steps, each focusing on specific components of the system.

## Project Overview

The project aims to build a self-supervised Vision-Transformer pipeline for morph-attack detection. It fine-tunes a masked-auto-encoder (MAE) backbone—either facebook/vit-mae-large (350M parameters) or Web-SSL MAE-1B (≈1B parameters)—to detect morphed face images across six different morphing styles (LMA, LMA-UBO, MIPGAN-I, MIPGAN-II, MorDiff and StyleGAN-IWBF).

### Project Structure

```
project_root/
│
├─ dataset/                # Raw images organized by type and morphing style
│   ├─ bonafide/
│   │   ├─ LMA/           # Each has P1/ and P2/ subfolders
│   │   ├─ LMA_UBO/
│   │   ├─ MIPGAN_I/
│   │   ├─ MIPGAN_II/
│   │   ├─ MorDiff/
│   │   └─ StyleGAN_IWBF/
│   └─ morph/
│       ├─ LMA/
│       ├─ LMA_UBO/
│       ├─ MIPGAN_I/
│       ├─ MIPGAN_II/
│       ├─ MorDiff/
│       └─ StyleGAN_IWBF/
│
├─ models/                 # Pre-trained model weights
│   ├─ vit_mae/           # ViT-MAE model files
│   │   ├─ config.json
│   │   ├─ preprocessor_config.json
│   │   └─ model.safetensors
│   └─ webssl/            # Web-SSL model files
│       ├─ config.json
│       ├─ preprocessor_config.json
│       └─ model.safetensors
│
├─ src/                    # Source code
│   ├─ config.py          # Configuration parameters
│   ├─ data_loader.py     # Data loading and augmentation
│   ├─ model.py           # Model architecture
│   ├─ train.py           # Training loop
│   ├─ test.py            # Evaluation
│   └─ utils.py           # Utility functions
│
├─ outputs/                # Generated outputs
│   ├─ models/            # Saved model weights
│   ├─ logs/              # Training and evaluation logs
│   └─ plots/             # Generated plots
│
├─ run_train.sh           # Script to run training
└─ run_test.sh            # Script to run evaluation
```

## Step 1: Configuration and Data Loading

In this step, we will create the configuration system and data loading pipeline.

### 1.1 Create `config.py`

The configuration file will contain all parameters needed for the project, organized into logical sections:

1. **Dataset Options**
   - `data_root`: Path to the dataset directory
   - `train_dataset`: Which morphing style to use for training (default: 'MorDiff')
   - `test_datasets`: List of all morphing styles for testing
   - `train_val_pct`: Percentage split between training and validation (70%)

2. **Training Hyperparameters**
   - `num_epochs`: Number of training epochs (10)
   - `batch_size`: Batch size for training (32)
   - `learning_rate`: Learning rate for optimizer (5e-5)
   - `recon_weight`: Weight for reconstruction loss (1 for MAE+classifier, 0 for supervised-only)
   - `lr_scheduler`: Learning rate scheduler ('cosine' or 'none')
   - `eta_min`: Minimum learning rate at the end of cosine schedule (0.0)
   - `T_max`: Number of epochs over which to decay (defaults to num_epochs)
   - `use_fixed_seed`: Whether to use a fixed seed (True) or random seed (False)
   - `seed`: Seed value when use_fixed_seed is True (e.g., 42), ignored when use_fixed_seed is False

3. **Regularization and Backbone Control**
   - `freeze_strategy`: Strategy for freezing backbone layers ('freeze_except_lastN', 'none', 'backbone_only', or 'gradual_unfreeze')
   - `warmup_epochs`: Number of epochs to train only the head (for 'gradual_unfreeze')
   - `mid_epochs`: Epoch to start unfreezing the full backbone (for 'gradual_unfreeze')
   - `freeze_lastN`: Number of final ViT blocks to keep trainable (2)
   - `dropout_p`: Dropout probability (0.5)
   - `use_mae_aug`: Whether to use MAE-specific augmentations (True)
   - `mae_mask_ratio`: Ratio of patches to mask in MAE (0.75, range: 0.0 to 1.0)
   - `use_cls_aug`: Whether to use classifier augmentations (True)

4. **Model Selection**
   - `pretrained_model`: Which pre-trained model to use ('vit_mae' or 'webssl')

5. **Output Configuration**
   - `final_model_name`: Template for saved model name
   - `log_metrics`: List of metrics to log
   - Output directory paths for models, logs, and plots

6. **Command-line Override System**
   - Implement a system to override configuration values via command-line arguments
   - Format: `--override key=value`

### 1.2 Create `data_loader.py`

The data loader will handle loading and preprocessing images from the dataset:

1. **Dataset Class**
   - Create a PyTorch Dataset class for morph detection
   - Handle loading images from file paths
   - Apply appropriate transformations
   - Return dictionary with image, label, and metadata

2. **File Path Gathering**
   - For training/validation:
     - Use P2 subset of the specified training dataset
     - Label bonafide images as 0, morph images as 1
     - Shuffle and split according to train_val_pct
   - For testing:
     - Use P1 subset of each test dataset
     - Maintain the same labeling convention

3. **Augmentation Pipelines**
   - Basic transforms (always applied):
     - Resize to 224×224 using Lanczos resampling
     - Convert to tensor and normalize
   - Classifier augmentations (optional):
     - Random rotation (±5°)
     - Color jitter (brightness, contrast, saturation)
     - Gaussian blur (10% probability)
     - Random resized crop (scale 0.9-1.1)
   - MAE augmentations:
     - Not handled in the data loader
     - Applied by the model during the forward pass
     - Controlled by two configuration parameters:
       - `use_mae_aug`: Whether to use MAE masking
         - When `True`: Model will mask patches and reconstruct them
         - When `False`: No masking will be applied
       - `mae_mask_ratio`: Ratio of patches to mask (default: 0.75)
         - Configurable between 0.0 and 1.0
         - Lower values (e.g., 0.4) make reconstruction easier
         - Higher values (e.g., 0.9) make reconstruction harder

4. **DataLoader Creation**
   - Create training loader with shuffling
   - Create validation loader without shuffling
   - Create separate test loaders for bonafide and morph images for each morphing style:
     - For each morphing style, create two separate loaders:
       - One for bonafide images from P1 subset
       - One for morph images from P1 subset
     - Process and evaluate each set separately
     - Save scores for each set separately

## Step 2: Model Architecture and Training

In this step, we will implement the model architecture and training pipeline.

### 2.1 Create `model.py`

The model file will define the architecture for morph detection:

1. **MorphDetector Class**
   - Initialize with configuration parameters
   - Load pre-trained MAE backbone (either ViT-MAE or WebSSL)
   - Add dropout and classification head on top of [CLS] token
   - Implement forward pass with both reconstruction and classification branches

2. **Backbone Loading**
   - Load model weights from local directory (models/vit_mae/ or models/webssl/)
   - Extract encoder part of the MAE model
   - Configure model according to the original config.json

3. **Classification Head**
   - Add Dropout layer with probability from config
   - Add Linear layer mapping from hidden dimension to 1
   - Add Sigmoid activation for binary classification

4. **Loss Functions**
   - MAE loss: Mean squared error on reconstructed patches
   - BCE loss: Binary cross-entropy for classification
   - Total loss: Weighted combination based on recon_weight

5. **Freezing Strategies**
   - Implement 'none' strategy: All layers trainable
   - Implement 'backbone_only' strategy: Freeze entire backbone
   - Implement 'freeze_except_lastN' strategy: Freeze all but last N transformer blocks
   - Implement 'gradual_unfreeze' strategy: Three-phase schedule:
     - Phase A (epochs 0 → warmup_epochs-1): backbone frozen, only head learns
     - Phase B (epochs warmup_epochs → mid_epochs-1): unfreeze last N blocks in addition to head
     - Phase C (epoch mid_epochs → end): unfreeze full backbone

6. **Forward Method**
   - Process input through encoder
   - Extract [CLS] token for classification
   - Apply dropout and classification head
   - Calculate reconstruction loss if recon_weight > 0
   - Return classification score and losses

### 2.2 Create `train.py`

The training script will handle the complete training workflow:

1. **Setup**
   - Parse command-line arguments and load configuration
   - Set seed for reproducibility:
     - If use_fixed_seed is True, use the specified seed value
     - If use_fixed_seed is False, generate a random seed and record it
   - Apply seed to all random number generators (Python, NumPy, PyTorch)
   - Create output directories
   - Build data loaders for training and validation
   - Initialize model and optimizer (AdamW)
   - Set up learning rate scheduler if enabled (cosine or none)
   - Set up metrics tracking

2. **Training Loop**
   - Iterate through epochs (1 to num_epochs)
   - Apply freezing strategy based on current epoch:
     - For 'gradual_unfreeze':
       - Epochs 0 to warmup_epochs-1: Only train classification head
       - Epochs warmup_epochs to mid_epochs-1: Unfreeze last N blocks
       - Epochs mid_epochs to end: Unfreeze full backbone
     - For other strategies: Apply once at the beginning
   - Apply learning rate scheduling if enabled:
     - For 'cosine': Update learning rate based on cosine schedule
     - For 'none': Keep initial learning rate
   - For each batch:
     - Forward pass through model
     - Calculate total loss
     - Backward pass and optimizer step
     - Track metrics

3. **Validation**
   - After each training epoch, evaluate on validation set
   - Calculate validation loss and accuracy
   - Track metrics for monitoring

4. **Logging**
   - Record metrics after each epoch
   - Save to CSV file in outputs/logs/
   - Write header row with column names
   - Include seed value in every row for traceability
   - Format: seed, epoch, train_loss, val_loss, val_acc, learning_rate
   - For the first row (epoch 0), include whether the seed was fixed or randomly generated

5. **Model Saving**
   - Save final model weights after training
   - Path: outputs/models/{train_dataset}_morphdetector.pt

6. **Visualization**
   - Generate loss and accuracy curves
   - Save plots to outputs/plots/

## Step 3: Evaluation and Utilities

In this step, we will implement the evaluation pipeline and utility functions.

### 3.1 Create `utils.py`

The utilities file will contain helper functions used across the project:

1. **Logging Functions**
   - Function to initialize CSV loggers
   - Function to append metrics to CSV files
   - Function to create timestamped filenames

2. **Plotting Functions**
   - Function to plot loss curves
   - Function to plot accuracy curves
   - Function to plot ROC curves
   - Function to create combined ROC plot for all datasets

3. **Metric Calculation**
   - Function to calculate accuracy
   - Function to calculate APCER (Attack Presentation Classification Error Rate)
   - Function to calculate BPCER (Bona Fide Presentation Classification Error Rate)
   - Function to calculate EER (Equal Error Rate)
   - Function to calculate ROC curve and AUC

4. **Model Utilities**
   - Function to save model weights
   - Function to load model weights
   - Function to apply freezing strategy

5. **Miscellaneous Helpers**
   - Function to ensure directories exist
   - Function to set seeds for reproducibility:
     - Handle both fixed and random seed options
     - Apply seed to Python, NumPy, and PyTorch random generators
     - Set CUDA deterministic mode for full reproducibility
     - Return the actual seed used (fixed or generated)
   - Function to format configuration for logging

### 3.2 Create `test.py`

The testing script will handle evaluation across multiple datasets:

1. **Setup**
   - Parse command-line arguments and load configuration
   - Use the same seed setting mechanism as in training:
     - If use_fixed_seed is True, use the specified seed value
     - If use_fixed_seed is False, generate a random seed and record it
   - Load trained model weights
   - Set model to evaluation mode

2. **Per-Dataset Evaluation**
   - For each dataset in test_datasets:
     - Build separate test loaders for bonafide and morph images
     - Process each set separately:
       - Run forward pass on all bonafide samples
       - Run forward pass on all morph samples
       - Collect raw scores and true labels for each set
     - Calculate metrics (accuracy, APCER, BPCER, EER, AUC)
     - Save results to CSV

3. **Results Logging**
   - Save per-sample results (dataset, label, score)
   - Save summary metrics for each dataset
   - Include seed value in all log files for traceability
   - Format: seed, dataset, accuracy, accuracy_bonafide, accuracy_morph, APCER, BPCER, EER, AUC
   - Record whether the seed was fixed or randomly generated

4. **Visualization**
   - Generate ROC curve for each dataset
   - Generate combined ROC curve with all datasets
   - Save plots to outputs/plots/

5. **Results Summary**
   - Print summary of results to console
   - Highlight best and worst performing datasets
   - Show average performance across all datasets

## Step 4: Shell Scripts and Final Integration

In this step, we will create shell scripts to run the pipeline and ensure all components work together.

### 4.1 Create `run_train.sh`

The training script will provide a simple interface to run the training pipeline:

1. **Environment Setup**
   - Set up environment variables if needed
   - Activate virtual environment if used

2. **Command Construction**
   - Build command to run train.py
   - Include configuration path
   - Allow for command-line overrides

3. **Execution**
   - Run the training command
   - Capture and display output
   - Show path to saved model

4. **Example Usage**
   - Include examples of common training scenarios:
     - Training with different datasets
     - Training with different freeze strategies
     - Training with different reconstruction weights
     - Training with different pre-trained models
     - Training with fixed vs. random seeds:
       ```bash
       # Fixed seed for reproducibility
       ./run_train.sh --override use_fixed_seed=True --override seed=42

       # Random seed for experimental variation
       ./run_train.sh --override use_fixed_seed=False
       ```

### 4.2 Create `run_test.sh`

The testing script will provide a simple interface to run the evaluation pipeline:

1. **Environment Setup**
   - Set up environment variables if needed
   - Activate virtual environment if used

2. **Command Construction**
   - Build command to run test.py
   - Include configuration path
   - Include path to trained model
   - Allow for command-line overrides

3. **Execution**
   - Run the testing command
   - Capture and display output
   - Show paths to result files and plots

4. **Example Usage**
   - Include examples of common testing scenarios:
     - Testing on all datasets
     - Testing on specific datasets
     - Testing with different thresholds
     - Testing with fixed vs. random seeds:
       ```bash
       # Use same fixed seed as training for consistent evaluation
       ./run_test.sh --override use_fixed_seed=True --override seed=42

       # Use random seed for robustness testing
       ./run_test.sh --override use_fixed_seed=False
       ```

### 4.3 Final Integration and Testing

Ensure all components work together correctly:

1. **Directory Structure**
   - Verify all required directories exist
   - Ensure correct permissions

2. **End-to-End Testing**
   - Run complete pipeline on a small subset of data
   - Verify all outputs are generated correctly
   - Check for any errors or warnings

3. **Documentation**
   - Add comments to all code files
   - Update README if needed
   - Document any known issues or limitations

4. **Optimization**
   - Profile code for bottlenecks
   - Optimize critical sections if needed
   - Ensure efficient use of GPU resources

## Detailed Implementation Notes

### Data Handling

- P2 folders are used for training and validation
- P1 folders are reserved for testing
- This separation ensures no identity leakage between train and test sets
- Images are resized to 224×224 using Lanczos resampling for best quality
- Augmentations are applied consistently to both reconstruction and classification branches

### Model Architecture Details

- The ViT encoder processes images as 16×16 patches
- The [CLS] token serves as the representation for classification
- Dropout is applied to the [CLS] token before classification
- The MAE decoder is used during training but discarded during inference
- The model can operate in pure classification mode (recon_weight=0) or joint mode (recon_weight>0)

### Training Dynamics

- AdamW optimizer is used with weight decay
- Learning rate is kept low (5e-5) for stable fine-tuning
- Cosine learning rate scheduling gradually reduces learning rate over training
- Freezing strategies control which parts of the backbone adapt
- Gradual unfreezing allows progressive adaptation from specific to general features
- Early layers capture generic features, later layers specialize for morph detection
- Dropout prevents overfitting to specific artifacts in the training domain
- Seed management ensures reproducibility:
  - Fixed seed option for exact reproducibility across runs
  - Random seed option with tracking for experimental variation
  - Seed value is recorded in all logs for traceability

### Evaluation Metrics

- Accuracy: Overall percentage of correct classifications
- APCER: Percentage of morphs misclassified as bonafide (false negatives)
- BPCER: Percentage of bonafide misclassified as morphs (false positives)
- EER: Operating point where APCER equals BPCER
- ROC AUC: Area under the ROC curve, measuring overall discrimination ability

### Cross-Dataset Generalization

- Training on one morphing style tests generalization to unseen styles
- Different GAN-based morphs have different artifacts
- The model must learn to detect the concept of "morphing" rather than specific artifacts
- Self-supervision helps maintain focus on global face structure
- Performance across datasets reveals which morphing techniques are hardest to detect

## Experimental Variations and Ablation Studies

The pipeline is designed to facilitate various experimental configurations to understand what factors contribute most to morph detection performance:

### 1. Backbone Comparison

Compare the performance between two pre-trained models:

- **ViT-MAE-Large (350M parameters)**
  - Pre-trained on ImageNet-1K with masked autoencoding
  - Well-established baseline for vision tasks
  - More computationally efficient

- **WebSSL-MAE-1B (1B parameters)**
  - Pre-trained on 2 billion web images
  - Potentially better generalization due to scale and diversity
  - More computationally demanding

### 2. Self-Supervision Impact

Evaluate the effect of the reconstruction objective:

- **Pure Classification (recon_weight=0)**
  - Only binary cross-entropy loss
  - Faster training
  - May overfit to specific artifacts

- **Joint Training (recon_weight=1)**
  - Combined reconstruction and classification losses
  - Regularizes the model through self-supervision
  - May improve generalization to unseen morphing techniques

### 3. Freezing Strategy Comparison

Assess different approaches to fine-tuning:

- **Full Fine-tuning (freeze_strategy='none')**
  - All layers adapt to the morph detection task
  - Maximum flexibility but risk of catastrophic forgetting
  - May overfit on small datasets

- **Frozen Backbone (freeze_strategy='backbone_only')**
  - Only the classification head is trained
  - Preserves pre-trained representations
  - Limited adaptation to the specific task

- **Partial Unfreezing (freeze_strategy='freeze_except_lastN')**
  - Early layers remain frozen, later layers adapt
  - Balance between stability and task-specific adaptation
  - Typically offers best generalization

- **Gradual Unfreezing (freeze_strategy='gradual_unfreeze')**
  - Three-phase training schedule:
    - Phase A: Only train the classification head (epochs 0 to warmup_epochs-1)
    - Phase B: Unfreeze last N blocks (epochs warmup_epochs to mid_epochs-1)
    - Phase C: Unfreeze entire backbone (epochs mid_epochs to end)
  - Allows progressive adaptation from specific to general features
  - Mitigates catastrophic forgetting while enabling full adaptation

### 4. Augmentation Ablation

Measure the contribution of different augmentation strategies:

- **No Augmentation**
  - Basic resizing only
  - Baseline performance

- **Classifier Augmentation Only (use_cls_aug=True, use_mae_aug=False)**
  - Standard photometric and geometric transformations
  - Improves robustness to image variations
  - No masked patch reconstruction

- **MAE Augmentation Only (use_cls_aug=False, use_mae_aug=True)**
  - Masked patch reconstruction during model forward pass
  - Model masks patches according to `mae_mask_ratio` (default: 75%)
  - Masking ratio can be adjusted (e.g., 40% for easier tasks, 90% for harder tasks)
  - Focuses on global structure understanding
  - No classifier-specific augmentations

- **Full Augmentation (use_cls_aug=True, use_mae_aug=True)**
  - Combined approach with both classifier augmentations and MAE masking
  - Classifier augmentations applied in data loader
  - MAE masking applied during model forward pass
  - Maximum regularization

### 5. Cross-Dataset Generalization

Analyze how training on one morphing style transfers to others:

- **Within-Domain Performance**
  - How well the model performs on the same morphing style it was trained on
  - Upper bound on expected performance

- **Cross-Domain Performance**
  - How well the model generalizes to unseen morphing styles
  - Measures true robustness of the approach

- **Hardest Domains**
  - Identify which morphing techniques are most challenging to detect
  - Guide future research directions

## Conclusion

This implementation plan provides a comprehensive framework for building a state-of-the-art morph detection system using self-supervised Vision Transformers. By following these detailed instructions, you will create a pipeline that:

1. Leverages powerful pre-trained vision models
2. Incorporates self-supervision for improved generalization
3. Provides flexible configuration for experimental variations
4. Generates detailed metrics and visualizations for analysis
5. Enables systematic evaluation across multiple morphing techniques

The modular design allows for easy extension to new datasets, backbones, or training strategies in the future. The comprehensive evaluation framework will provide insights into which approaches work best for this challenging security application.
