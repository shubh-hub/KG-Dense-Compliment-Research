#!/bin/bash

# Comprehensive dataset fixing and analysis
# Fixes Natural Questions, Wikidata5M, and performs meticulous analysis

set -e

echo "=========================================="
echo "Comprehensive Dataset Fixing & Analysis"
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

# Run comprehensive fixing and analysis
echo "Starting comprehensive dataset fixing and analysis..."
echo "This will:"
echo "1. Fix Natural Questions dataset format issues"
echo "2. Re-download and fix Wikidata5M corruption"
echo "3. Perform meticulous data analysis across all datasets"
echo "4. Generate comprehensive quality reports"
echo ""

python fix_datasets_comprehensive.py

echo ""
echo "Comprehensive dataset processing completed!"
echo "Check logs/comprehensive_data_analysis.log for detailed logs"
echo "Check data/analysis/ for analysis reports"
