#!/bin/bash

# Natural Questions Download Script
echo "Starting Natural Questions download and preprocessing..."

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

# Initialize conda
if [ -f "$HOME/miniforge3/bin/conda" ]; then
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
fi

# Activate environment
conda activate kg-dense

# Run the download script
python scripts/data_processing/download_natural_questions.py

echo "Natural Questions download completed!"
