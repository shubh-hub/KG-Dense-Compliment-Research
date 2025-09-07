#!/bin/bash

# Optimized Data Processing Pipeline for KG + Dense Vector Complementarity Research
# Processes 76.32GB of datasets efficiently on Apple M1 MacBook Air
# Based on detailed analysis: 30-45 minutes expected processing time

echo "=========================================="
echo "OPTIMIZED DATA PROCESSING PIPELINE"
echo "Processing 76.32GB of research datasets"
echo "=========================================="

# Set M1 optimization environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Memory management for M1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Activate conda environment
echo "Activating kg-dense environment..."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate kg-dense

# Check if environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "kg-dense" ]]; then
    echo "❌ Failed to activate kg-dense environment"
    exit 1
fi

echo "✅ Environment activated: $CONDA_DEFAULT_ENV"

# Check available memory
echo "Checking system resources..."
memory_gb=$(sysctl hw.memsize | awk '{print int($2/1024/1024/1024)}')
echo "Available RAM: ${memory_gb}GB"

if [ "$memory_gb" -lt 8 ]; then
    echo "⚠️  Low memory detected. Processing will use conservative settings."
fi

# Create processing log
log_dir="logs/processing"
mkdir -p "$log_dir"
log_file="$log_dir/optimized_processing_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Logging to: $log_file"

# Run the optimized processing pipeline
echo "Starting optimized data processing..."
echo "Expected time: 30-45 minutes"
echo "Processing order: BEIR IR → MS MARCO QA → Wikidata5M KG → Natural Questions QA"

python optimized_data_processor.py 2>&1 | tee "$log_file"

# Check if processing was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Optimized data processing completed successfully!"
    
    # Verify processed data
    echo "Verifying processed datasets..."
    
    processed_dir="data/processed"
    
    if [ -d "$processed_dir/ir" ] && [ -d "$processed_dir/qa" ] && [ -d "$processed_dir/kg" ]; then
        echo "✅ All dataset directories created"
        
        # Count processed files
        ir_files=$(find "$processed_dir/ir" -name "*.json" | wc -l)
        qa_files=$(find "$processed_dir/qa" -name "*.json" | wc -l)
        kg_files=$(find "$processed_dir/kg" -name "*.json" | wc -l)
        
        echo "📊 Processed files: IR($ir_files), QA($qa_files), KG($kg_files)"
        
        # Calculate total processed size
        processed_size=$(du -sh "$processed_dir" | cut -f1)
        echo "📦 Total processed size: $processed_size"
        
        echo "=========================================="
        echo "🎉 DATA PROCESSING PIPELINE COMPLETE!"
        echo "=========================================="
        echo "All datasets are now ready for complementarity research"
        echo ""
        echo "Next steps:"
        echo "1. Validate processed data: python analyze_existing_data.py"
        echo "2. Start training: python scripts/training/train_models.py --experiment_name complementarity_baseline"
        echo "3. Monitor experiments: check logs/training/ for progress"
        echo "=========================================="
        
    else
        echo "⚠️  Some processed directories missing. Check log for details."
        exit 1
    fi
    
else
    echo "❌ Data processing failed"
    echo "Check log file: $log_file"
    exit 1
fi
