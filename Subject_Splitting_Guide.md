# Subject-Based Splitting Guide

This document explains the subject-based splitting feature in the SSL_MAE codebase, why it's important for preventing data leakage, and how to use it.

## The Problem: Data Leakage in Morph Detection

In morph detection tasks, data leakage can occur when images of the same subject appear in both training and validation sets. This is particularly problematic because:

1. **Subject-specific features**: The model may learn to recognize specific subjects rather than generalizing to detect morphs.
2. **Unrealistic performance**: This leads to artificially high validation accuracy and unrealistically low validation loss.
3. **Poor generalization**: The model will perform poorly on new subjects not seen during training.

### Example of Data Leakage

Consider these file naming patterns:

- Bonafide images: `04376-1.jpg`, `04376-2.jpg`, etc.
- Morph images: `04376d103-vs-04379d116.jpg`

With random splitting, images of subject `04376` could end up in both training and validation sets. The model then "cheats" by recognizing this specific subject rather than learning generalizable features for morph detection.

## The Solution: Subject-Based Splitting

Subject-based splitting ensures that all images of a particular subject (both bonafide and morphs) are either entirely in the training set or entirely in the validation set. This prevents the model from using subject-specific features to "cheat" on the validation set.

### How It Works

1. **Extract subject IDs**: The system extracts subject IDs from all filenames.
2. **Split subjects**: Unique subject IDs are split into training and validation groups.
3. **Assign images**: Images are assigned to training or validation based on their subject IDs.
4. **Handle morphs conservatively**: For morph images (which contain two subjects), the image is assigned to validation if either subject is in the validation set.

## Enabling Subject-Based Splitting

Subject-based splitting is disabled by default but can be easily enabled:

### Via Command Line

```bash
python src/train.py --override use_subject_splitting=True
```

### In Configuration File

Edit `src/config.py` and set:

```python
self.use_subject_splitting = True
```

## Expected Results

When you enable subject-based splitting, you should see:

1. **More realistic validation metrics**:
   - Validation accuracy will be lower (not 1.0)
   - Validation loss will be higher (not 0.0)

2. **Better generalization**:
   - The model will perform better on completely new subjects
   - Performance on the test set will be closer to validation performance

## Implementation Details

The subject-based splitting is implemented in `src/subject_splitter.py` and includes:

1. **Subject ID extraction**: Extracts subject IDs from filenames using pattern recognition.
2. **Conservative morph handling**: For morph images, both subject IDs are considered.
3. **Detailed statistics**: Prints statistics about the split to help you understand the data distribution.

## Troubleshooting

If you encounter issues with subject-based splitting:

1. **Check file naming**: The system expects specific file naming patterns. If your files follow a different pattern, you may need to modify the `extract_subject_id` function in `subject_splitter.py`.

2. **Class imbalance**: Subject-based splitting may result in class imbalance. The balanced sampler should help address this, but you may need to adjust class weights.

3. **Small validation set**: If you have few subjects, the validation set may be small. Consider using cross-validation with subject-based splitting for more reliable performance estimates.

## Conclusion

Subject-based splitting is essential for realistic performance evaluation in morph detection tasks. By preventing data leakage, it helps you build models that truly generalize to new subjects rather than memorizing specific individuals in your training data.

Enable subject-based splitting to get a more accurate picture of your model's true performance and to build models that generalize better to unseen data.
