#!/bin/bash

# M1 MacBook Environment Verification Runner
# Fixes OpenMP conflicts and sets up proper conda environment

echo "Setting up environment for M1 MacBook verification..."

# Fix OpenMP library conflict (common on M1 Macs)
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# Initialize conda if not in PATH
if ! command -v conda &> /dev/null; then
    echo "Conda not found in PATH. Attempting to initialize..."
    
    # Common conda installation paths
    CONDA_PATHS=(
        "$HOME/miniforge3/bin/conda"
        "$HOME/miniconda3/bin/conda"
        "$HOME/anaconda3/bin/conda"
        "/opt/homebrew/Caskroom/miniforge/base/bin/conda"
        "/usr/local/Caskroom/miniforge/base/bin/conda"
    )
    
    for conda_path in "${CONDA_PATHS[@]}"; do
        if [ -f "$conda_path" ]; then
            echo "Found conda at: $conda_path"
            # Initialize conda for this session
            eval "$($conda_path shell.bash hook)"
            break
        fi
    done
    
    # If still not found, try to source conda initialization
    if ! command -v conda &> /dev/null; then
        for init_script in "$HOME/miniforge3/etc/profile.d/conda.sh" \
                          "$HOME/miniconda3/etc/profile.d/conda.sh" \
                          "$HOME/anaconda3/etc/profile.d/conda.sh"; do
            if [ -f "$init_script" ]; then
                echo "Sourcing conda initialization: $init_script"
                source "$init_script"
                break
            fi
        done
    fi
fi

# Check if conda is now available
if command -v conda &> /dev/null; then
    echo "✓ Conda found and initialized"
    
    # Try to activate kg-dense environment
    if conda env list | grep -q "kg-dense"; then
        echo "Activating kg-dense environment..."
        conda activate kg-dense
    else
        echo "⚠ kg-dense environment not found. Using base environment."
        echo "You may need to create the environment first with:"
        echo "  conda create -n kg-dense python=3.9"
        echo "  conda activate kg-dense"
    fi
else
    echo "⚠ Conda still not available. Proceeding with system Python."
fi

echo ""
echo "Environment variables set:"
echo "  KMP_DUPLICATE_LIB_OK=$KMP_DUPLICATE_LIB_OK"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""

# Install missing dependency first
pip install importlib_metadata

# Run the verification script
echo "Running verification script..."
python3 verify_setup.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "🎉 Verification completed successfully!"
else
    echo ""
    echo "❌ Verification failed with exit code: $exit_code"
    echo ""
    echo "Common fixes:"
    echo "1. Install/activate conda environment: conda activate kg-dense"
    echo "2. Install missing packages: pip install torch torchvision torch-geometric"
    echo "3. If OpenMP errors persist, restart terminal and try again"
fi

exit $exit_code
