#!/bin/bash

# Fix NumPy Import Issues
echo "Fixing numpy import issues..."

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

# Initialize conda
if [ -f "$HOME/miniforge3/bin/conda" ]; then
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
fi

# Activate environment
conda activate kg-dense

echo "Reinstalling numpy and dependent packages..."

# Uninstall packages that depend on numpy
pip uninstall -y datasets pyarrow numpy

# Reinstall numpy first
pip install numpy==1.24.4

# Then reinstall pyarrow and datasets
pip install pyarrow==14.0.1
pip install datasets==2.14.6

echo "Testing the fix..."
python3 -c "
import numpy as np
print('✓ numpy:', np.__version__)

import datasets
print('✓ datasets:', datasets.__version__)

import accelerate
print('✓ accelerate:', accelerate.__version__)
"

echo "NumPy fix completed!"
