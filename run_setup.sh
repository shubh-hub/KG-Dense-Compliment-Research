#!/bin/bash

# Run Data Structure Setup with proper environment
echo "Setting up data structure for KG + Dense Vector Research..."

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

# Initialize conda
if [ -f "$HOME/miniforge3/bin/conda" ]; then
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
fi

# Activate environment
conda activate kg-dense

# Run setup
python3 setup_data_structure.py

echo "Setup complete!"
