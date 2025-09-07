#!/bin/bash

# Process full datasets for SOTA-level research
# No downsampling - maintains dataset integrity for publication-quality results

set -e

echo "=========================================="
echo "Full Dataset Processing for SOTA Research"
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

# Run full dataset processing
echo "Processing full datasets (no downsampling)..."
echo "This maintains SOTA research integrity and benchmark validity"
echo ""

python process_full_datasets.py

echo ""
echo "Full dataset processing completed!"
echo "Ready for SOTA-level model development and evaluation"
