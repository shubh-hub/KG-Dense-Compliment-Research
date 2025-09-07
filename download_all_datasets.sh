#!/bin/bash

# Master Dataset Download Script
# Downloads all datasets for KG + Dense Vector Research

set -e  # Exit on any error

echo "Starting dataset download process..."
echo "This will download approximately 50GB of data"
echo "Ensure you have sufficient disk space and internet bandwidth"
echo ""

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

# Initialize conda
if [ -f "$HOME/miniforge3/bin/conda" ]; then
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
fi

# Activate environment
conda activate kg-dense

# Create necessary directories
mkdir -p data/raw data/processed logs/preprocessing

# Download datasets in order of priority
echo "=== Phase 1: QA Datasets ==="
python scripts/data_processing/download_natural_questions.py
python scripts/data_processing/download_ms_marco_qa.py

echo "=== Phase 2: IR Datasets ==="
python scripts/data_processing/download_ms_marco_passage.py
python scripts/data_processing/download_beir.py

echo "=== Phase 3: KG Dataset ==="
python scripts/data_processing/download_wikidata5m.py

echo "=== Download Complete ==="
echo "All datasets downloaded successfully!"
echo "Next: Run preprocessing scripts"
