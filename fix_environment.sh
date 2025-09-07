#!/bin/bash

# Fix Environment Issues Script
# Resolves scipy/sklearn compatibility problems on M1 MacBook

echo "Fixing environment issues..."

# Set OpenMP fix
export KMP_DUPLICATE_LIB_OK=TRUE

# Initialize conda
if [ -f "$HOME/miniforge3/bin/conda" ]; then
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
fi

# Activate environment
conda activate kg-dense

echo "Current environment: $(conda info --envs | grep '*')"

# Fix scipy/sklearn compatibility issue
echo "Fixing scipy/sklearn compatibility..."

# Uninstall problematic packages
pip uninstall -y scipy scikit-learn

# Reinstall with compatible versions
echo "Installing compatible scipy and scikit-learn versions..."
pip install scipy==1.11.4
pip install scikit-learn==1.3.2

# Also ensure matplotlib and seaborn are properly installed
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
pip install pandas==2.1.4
pip install numpy==1.24.4

echo "Package fixes completed!"

# Test the fixes
echo "Testing scipy import..."
python3 -c "import scipy; print(f'✓ scipy {scipy.__version__}')" || echo "✗ scipy still failing"

echo "Testing sklearn import..."
python3 -c "import sklearn; print(f'✓ sklearn {sklearn.__version__}')" || echo "✗ sklearn still failing"

echo "Testing transformers functionality..."
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokens = tokenizer('test', return_tensors='pt')
print('✓ transformers working')
" || echo "✗ transformers still failing"

echo ""
echo "Environment fix completed. Run verification again with:"
echo "./run_verification.sh"
