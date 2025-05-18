# Folder Structure Update

## Overview

This document describes the changes made to the codebase to align with the actual folder structure used in the datasets.

## Changes Made

The codebase was previously using "P1" and "P2" as folder names for testing and training data respectively, but the actual folder structure in the datasets uses "test" and "train" folders. The following changes were made to align the code with the actual folder structure:

1. Updated the `get_file_paths` function in `src/data_loader.py` to use "train" as the default value for the `subset` parameter (this was already done).

2. Updated the `create_test_loaders` function in `src/data_loader.py` to use "test" instead of "P1" for the testing subset.

3. Kept the `create_data_loaders` function in `src/data_loader.py` using "train" for the training subset (this was already done).

## Folder Structure

The current folder structure is:

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

Where `{dataset_name}` is one of the following:
- LMA
- LMA_UBO
- MIPGAN_I
- MIPGAN_II
- MorDiff
- StyleGAN_IWBF
- etc.

## Benefits

This change makes the codebase more intuitive and easier to understand, as the folder names now directly reflect their purpose (training or testing) rather than using abstract names like "P1" and "P2".
