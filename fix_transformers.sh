#!/bin/bash

# Fix Transformers/Accelerate Circular Import Issue
# Install compatible versions that work together

echo "Fixing transformers/accelerate compatibility..."

# Set OpenMP fix
export KMP_DUPLICATE_LIB_OK=TRUE

# Initialize conda
if [ -f "$HOME/miniforge3/bin/conda" ]; then
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
fi

# Activate environment
conda activate kg-dense

echo "Uninstalling problematic packages..."
pip uninstall -y transformers accelerate

echo "Installing compatible versions..."
pip install transformers==4.35.2
pip install accelerate==0.24.1

echo "Testing the fix..."
python3 -c "
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokens = tokenizer('test', return_tensors='pt')
    print('✓ transformers working with compatible versions')
except Exception as e:
    print(f'✗ still failing: {e}')
"

echo "Fix completed. Run verification again with: ./run_verification.sh"
