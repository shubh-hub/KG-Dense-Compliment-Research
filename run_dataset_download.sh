#!/bin/bash

# Download remaining datasets for KG + Dense Vector research
# Total: ~7.65GB (BEIR + Wikidata5M + MS MARCO QA + MS MARCO Passage)

set -e

echo "=========================================="
echo "KG + Dense Vector Research Dataset Download"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Fix OpenMP library conflicts on M1 Mac
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

# Initialize conda if not already done
if ! command -v conda &> /dev/null; then
    echo "Initializing conda..."
    source ~/miniforge3/etc/profile.d/conda.sh
fi

# Activate environment
echo "Activating kg-dense environment..."
conda activate kg-dense

# Verify environment
echo "Verifying environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"

# Create logs directory
mkdir -p logs

# Run download script
echo "Starting dataset downloads..."
echo "Order: BEIR (~0.15GB) → Wikidata5M (1.8GB) → MS MARCO QA (2.5GB) → MS MARCO Passage (3.2GB)"
echo ""

python download_remaining_datasets.py

echo ""
echo "Dataset download completed!"
echo "Check logs/dataset_download.log for detailed logs"
