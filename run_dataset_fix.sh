#!/bin/bash

# Comprehensive Dataset Fixing Script for KG + Dense Vector Research
# Fixes Wikidata5M, Natural Questions, and BEIR datasets

set -e  # Exit on any error

echo "============================================================"
echo "COMPREHENSIVE DATASET FIXING FOR KG + DENSE VECTOR RESEARCH"
echo "============================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables for M1 compatibility
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Initialize conda
echo "Initializing conda environment..."
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "Conda not found. Trying to initialize from common locations..."
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh" ]; then
        source "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh"
    else
        echo "ERROR: Could not find conda installation"
        exit 1
    fi
    eval "$(conda shell.bash hook)"
fi

# Activate kg-dense environment
echo "Activating kg-dense environment..."
if conda activate kg-dense; then
    echo "✓ Successfully activated kg-dense environment"
else
    echo "ERROR: Failed to activate kg-dense environment"
    echo "Please ensure the environment exists: conda create -n kg-dense python=3.9"
    exit 1
fi

# Verify Python environment
echo "Verifying Python environment..."
python --version
echo "Python path: $(which python)"

# Create logs directory
mkdir -p logs

# Run comprehensive dataset fixing
echo ""
echo "Starting comprehensive dataset fixing..."
echo "This will fix:"
echo "  1. Wikidata5M KG dataset (CRITICAL)"
echo "  2. Natural Questions QA dataset" 
echo "  3. BEIR IR benchmark datasets"
echo "  4. Generate comprehensive analysis"
echo ""

# Run the fixing script with logging
if python fix_all_datasets.py 2>&1 | tee logs/dataset_fixing_$(date +%Y%m%d_%H%M%S).log; then
    echo ""
    echo "============================================================"
    echo "✅ DATASET FIXING COMPLETED SUCCESSFULLY!"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Review the analysis report in data/analysis/"
    echo "  2. Verify all datasets are ready for research"
    echo "  3. Begin model architecture development"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "❌ DATASET FIXING ENCOUNTERED ERRORS"
    echo "============================================================"
    echo ""
    echo "Check the log file for details:"
    echo "  - logs/dataset_fixing_*.log"
    echo "  - logs/comprehensive_data_analysis.log"
    echo ""
    echo "You may need to:"
    echo "  1. Check internet connection"
    echo "  2. Verify disk space availability"
    echo "  3. Install missing dependencies"
    echo ""
    exit 1
fi

echo "Dataset fixing script completed."
