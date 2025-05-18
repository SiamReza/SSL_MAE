#!/bin/bash
# Testing script for the Self-Supervised Vision-Transformer Pipeline for Morph-Attack Detection

# Environment setup
# Uncomment and modify the following lines if you're using a virtual environment
# export PYTHONPATH=$PYTHONPATH:$(pwd)
# source venv/bin/activate

# Default values
CONFIG_PATH="src/config.py"
MODEL_PATH=""
DATASETS=""
THRESHOLD="0.5"
ARGS=""

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)
            MODEL_PATH="$2"
            ARGS="$ARGS --model_path $2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            ARGS="$ARGS --datasets $2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            ARGS="$ARGS --threshold $2"
            shift 2
            ;;
        --override)
            ARGS="$ARGS --override $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_test.sh [--model_path path] [--datasets dataset1 dataset2 ...] [--threshold value] [--override key=value]"
            exit 1
            ;;
    esac
done

# Create output directories if they don't exist
mkdir -p output/logs
mkdir -p output/plots

# Run the testing script
echo "Starting evaluation with arguments: $ARGS"
python src/test.py $ARGS

# Display paths to results
echo "Evaluation completed. Results saved to output/logs"
echo "Check output/plots for visualizations."

# Example usage (commented out):
#
# Test on all datasets:
# ./run_test.sh
#
# Test on specific datasets:
# ./run_test.sh --datasets LMA MIPGAN_I
#
# Test with a specific model:
# ./run_test.sh --model_path output/models/MorDiff_morphdetector.pt
#
# Test with a different threshold:
# ./run_test.sh --threshold 0.7
#
# Test with fixed seed for consistent evaluation:
# ./run_test.sh --override use_fixed_seed=True --override seed=42
#
# Test with random seed for robustness testing:
# ./run_test.sh --override use_fixed_seed=False
