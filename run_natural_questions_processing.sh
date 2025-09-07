#!/bin/bash

# Natural Questions Processing Script for M1 MacBook Air
# Processes 57GB of Natural Questions Parquet files with memory optimization

set -e

echo "=========================================="
echo "Natural Questions Processing Pipeline"
echo "=========================================="

# Set M1 optimization environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Activate conda environment
echo "Activating kg-dense environment..."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate kg-dense

# Check environment
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs/processing

# Run Natural Questions processor
echo "Starting Natural Questions processing..."
echo "Processing 57GB of Parquet files..."

python natural_questions_processor.py

echo "Natural Questions processing completed!"
echo "Check data/processed/qa/ for output files"
echo "Check logs/processing/natural_questions_processing.log for detailed logs"
