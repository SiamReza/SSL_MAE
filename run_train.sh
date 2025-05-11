#!/bin/bash
# Training script for the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection

# Environment setup
# Uncomment and modify the following lines if you're using a virtual environment
# export PYTHONPATH=$PYTHONPATH:$(pwd)
# source venv/bin/activate

# Default values
CONFIG_PATH="src/config.py"
ARGS=""

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --override)
            ARGS="$ARGS --override $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_train.sh [--override key=value]"
            exit 1
            ;;
    esac
done

# Create output directories if they don't exist
mkdir -p output/models
mkdir -p output/logs
mkdir -p output/plots

# Run the training script
echo "Starting training with arguments: $ARGS"
python src/train.py $ARGS

# Display path to saved model
echo "Training completed. Model saved to output/models/{train_dataset}_morphdetector.pt"
echo "Check output/logs for training metrics and output/plots for visualizations."

# Example usage (commented out):
# 
# Train with different dataset:
# ./run_train.sh --override train_dataset=LMA
#
# Train with different freeze strategy:
# ./run_train.sh --override freeze_strategy=none
#
# Train with different reconstruction weight:
# ./run_train.sh --override recon_weight=0
#
# Train with different pre-trained model:
# ./run_train.sh --override pretrained_model=webssl
#
# Train with fixed seed for reproducibility:
# ./run_train.sh --override use_fixed_seed=True --override seed=42
#
# Train with random seed for experimental variation:
# ./run_train.sh --override use_fixed_seed=False
