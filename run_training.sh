#!/bin/bash

# Training Pipeline Runner for KG + Dense Vector Research
# Optimized for Apple M1 MacBook Air

set -e  # Exit on any error

echo "=============================================="
echo "KG + Dense Vector Training Pipeline"
echo "=============================================="

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Environment setup for M1 MacBook
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Activate conda environment
echo "Activating conda environment..."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate kg-dense

# Verify environment
echo "Verifying environment..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Create necessary directories
mkdir -p logs/training
mkdir -p results/experiments

# Set up logging
LOG_FILE="logs/training/training_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

# Function to run training with error handling
run_training() {
    local experiment_name="$1"
    local model_size="$2"
    local fusion_method="$3"
    local epochs="$4"
    local batch_size="$5"
    
    echo "----------------------------------------"
    echo "Running experiment: $experiment_name"
    echo "Model size: $model_size"
    echo "Fusion method: $fusion_method"
    echo "Epochs: $epochs"
    echo "Batch size: $batch_size"
    echo "----------------------------------------"
    
    python scripts/training/train_models.py \
        --experiment_name "$experiment_name" \
        --model_size "$model_size" \
        --fusion_method "$fusion_method" \
        --num_epochs "$epochs" \
        --batch_size "$batch_size" \
        --learning_rate 1e-4 \
        2>&1 | tee -a "$LOG_FILE"
    
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "✓ Experiment $experiment_name completed successfully"
    else
        echo "✗ Experiment $experiment_name failed with exit code $exit_code"
        return $exit_code
    fi
}

# Check if datasets are ready
echo "Checking dataset availability..."
if [ ! -d "data/processed" ]; then
    echo "Warning: Processed data directory not found. Creating sample data..."
    python create_sample_datasets.py 2>&1 | tee -a "$LOG_FILE"
fi

# Test data loading first
echo "Testing data loading..."
python -c "
import sys
sys.path.append('.')
from data_processing.data_loaders import create_data_processor, DataConfig

try:
    config = DataConfig.lightweight_config()
    processor = create_data_processor(config)
    dataloaders = processor.create_dataloaders(batch_size=4, num_workers=1)
    print(f'✓ Data loading successful. Available dataloaders: {list(dataloaders.keys())}')
except Exception as e:
    print(f'✗ Data loading failed: {e}')
    print('This is expected if datasets are still processing...')
" 2>&1 | tee -a "$LOG_FILE"

# Main training experiments
echo "Starting training experiments..."

# Experiment 1: Lightweight baseline with cross-attention fusion
if run_training "baseline_cross_attention" "lightweight" "cross_attention" 5 4; then
    echo "✓ Baseline cross-attention experiment completed"
else
    echo "⚠ Baseline cross-attention experiment failed, continuing..."
fi

# Experiment 2: Lightweight baseline with hierarchical gating fusion
if run_training "baseline_hierarchical_gating" "lightweight" "hierarchical_gating" 5 4; then
    echo "✓ Baseline hierarchical gating experiment completed"
else
    echo "⚠ Baseline hierarchical gating experiment failed, continuing..."
fi

# Quick model architecture test
echo "Testing model architectures..."
python -c "
import sys
sys.path.append('.')
import torch
from models.baseline_architectures import create_dense_model, create_kg_model, create_fusion_model

try:
    # Test model creation
    dense_model = create_dense_model('dense_only', 'lightweight', num_classes=2)
    kg_model = create_kg_model('kg_only', 'lightweight', num_entities=100, num_relations=10, num_classes=2)
    fusion_model = create_fusion_model('fusion', 'lightweight', 'cross_attention', 
                                     num_entities=100, num_relations=10, num_classes=2)
    
    print(f'✓ Dense model parameters: {sum(p.numel() for p in dense_model.parameters()):,}')
    print(f'✓ KG model parameters: {sum(p.numel() for p in kg_model.parameters()):,}')
    print(f'✓ Fusion model parameters: {sum(p.numel() for p in fusion_model.parameters()):,}')
    
    # Test forward pass with dummy data
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    dense_model = dense_model.to(device)
    kg_model = kg_model.to(device)
    fusion_model = fusion_model.to(device)
    
    # Dummy inputs
    input_ids = torch.randint(0, 1000, (2, 128)).to(device)
    attention_mask = torch.ones(2, 128).to(device)
    entity_ids = torch.randint(0, 100, (2, 10)).to(device)
    edge_index = torch.tensor([[0, 1], [1, 0]]).t().to(device)
    edge_type = torch.tensor([0, 0]).to(device)
    
    # Test forward passes
    dense_out = dense_model(input_ids, attention_mask)
    kg_out = kg_model(entity_ids, edge_index, edge_type)
    fusion_out = fusion_model(input_ids, attention_mask, entity_ids, edge_index, edge_type)
    
    print(f'✓ Dense output shape: {dense_out.shape}')
    print(f'✓ KG output shape: {kg_out.shape}')
    print(f'✓ Fusion output shape: {fusion_out.shape}')
    print('✓ All model architectures working correctly!')
    
except Exception as e:
    print(f'✗ Model architecture test failed: {e}')
    import traceback
    traceback.print_exc()
" 2>&1 | tee -a "$LOG_FILE"

# Test experiment tracking
echo "Testing experiment tracking..."
python -c "
import sys
sys.path.append('.')
from utils.experiment_tracker import create_experiment_tracker

try:
    tracker = create_experiment_tracker('test_tracking', {'test': True})
    tracker.log_metric('test_metric', 0.85)
    tracker.log_message('Test message')
    tracker.finish()
    print('✓ Experiment tracking working correctly!')
except Exception as e:
    print(f'✗ Experiment tracking test failed: {e}')
" 2>&1 | tee -a "$LOG_FILE"

# Summary
echo "=============================================="
echo "Training Pipeline Summary"
echo "=============================================="
echo "Log file: $LOG_FILE"
echo "Results directory: results/experiments/"

# Check if any experiments were successful
if ls results/experiments/ > /dev/null 2>&1; then
    echo "Experiment directories created:"
    ls -la results/experiments/ | grep "^d" | tail -5
else
    echo "No experiment results found - this is expected if datasets are still processing"
fi

echo ""
echo "Next steps:"
echo "1. Wait for dataset processing to complete"
echo "2. Run this script again to train models"
echo "3. Check results/experiments/ for training outputs"
echo "4. Use evaluation framework to analyze complementarity"

echo "=============================================="
echo "Training pipeline setup completed!"
echo "=============================================="
