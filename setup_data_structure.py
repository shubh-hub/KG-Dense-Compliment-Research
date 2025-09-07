#!/usr/bin/env python3
"""
Data Structure Setup for KG + Dense Vector Research
Creates organized directory structure and initial configuration
"""

import os
import json
from pathlib import Path

def create_data_structure(base_path):
    """Create organized directory structure for research data"""
    
    # Define directory structure
    directories = {
        'data': {
            'raw': {
                'qa': {
                    'natural_questions': {},
                    'ms_marco_qa': {}
                },
                'ir': {
                    'ms_marco_passage': {},
                    'beir': {
                        'nfcorpus': {},
                        'scifact': {},
                        'arguana': {},
                        'fiqa': {}
                    }
                },
                'kg': {
                    'wikidata5m': {}
                }
            },
            'processed': {
                'qa': {
                    'natural_questions': {},
                    'ms_marco_qa': {}
                },
                'ir': {
                    'ms_marco_passage': {},
                    'beir': {}
                },
                'kg': {
                    'wikidata5m': {}
                }
            },
            'embeddings': {
                'dense': {},
                'kg': {}
            },
            'splits': {
                'train': {},
                'dev': {},
                'test': {}
            }
        },
        'models': {
            'checkpoints': {},
            'pretrained': {},
            'fusion': {}
        },
        'results': {
            'experiments': {},
            'evaluations': {},
            'ablations': {}
        },
        'logs': {
            'training': {},
            'evaluation': {},
            'preprocessing': {}
        },
        'scripts': {
            'data_processing': {},
            'training': {},
            'evaluation': {}
        }
    }
    
    def create_dirs(structure, current_path):
        """Recursively create directory structure"""
        for name, subdirs in structure.items():
            dir_path = current_path / name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {dir_path}")
            
            if subdirs:
                create_dirs(subdirs, dir_path)
    
    base_path = Path(base_path)
    create_dirs(directories, base_path)
    
    return base_path

def create_dataset_config(base_path):
    """Create configuration file for dataset management"""
    
    config = {
        "datasets": {
            "qa": {
                "natural_questions": {
                    "name": "Natural Questions",
                    "task": "question_answering",
                    "source": "google-research-datasets/natural-questions",
                    "splits": ["train", "dev"],
                    "size_gb": 42.0,
                    "samples": {
                        "train": 307373,
                        "dev": 7830
                    },
                    "description": "Real questions from Google search with Wikipedia answers"
                },
                "ms_marco_qa": {
                    "name": "MS MARCO QA",
                    "task": "question_answering", 
                    "source": "microsoft/ms_marco",
                    "splits": ["train", "dev", "test"],
                    "size_gb": 2.5,
                    "samples": {
                        "train": 808731,
                        "dev": 101093,
                        "test": 101092
                    },
                    "description": "Questions from Bing search logs with passage answers"
                }
            },
            "ir": {
                "ms_marco_passage": {
                    "name": "MS MARCO Passage",
                    "task": "information_retrieval",
                    "source": "microsoft/ms_marco",
                    "splits": ["train", "dev", "test"],
                    "size_gb": 3.2,
                    "samples": {
                        "queries_train": 502939,
                        "queries_dev": 6980,
                        "passages": 8841823
                    },
                    "description": "Passage retrieval dataset from Bing search logs"
                },
                "beir": {
                    "name": "BEIR Benchmark",
                    "task": "information_retrieval",
                    "source": "BeIR/beir",
                    "datasets": {
                        "nfcorpus": {"size_gb": 0.03, "queries": 323, "docs": 3633},
                        "scifact": {"size_gb": 0.01, "queries": 300, "docs": 5183},
                        "arguana": {"size_gb": 0.01, "queries": 1406, "docs": 8674},
                        "fiqa": {"size_gb": 0.1, "queries": 648, "docs": 57638}
                    },
                    "description": "Diverse IR evaluation benchmark"
                }
            },
            "kg": {
                "wikidata5m": {
                    "name": "Wikidata5M",
                    "task": "knowledge_graph",
                    "source": "deepmind/wikidata5m",
                    "size_gb": 1.8,
                    "entities": 4594485,
                    "relations": 822,
                    "triples": 20624239,
                    "description": "Large-scale knowledge graph from Wikidata"
                }
            }
        },
        "preprocessing": {
            "max_seq_length": 512,
            "max_query_length": 64,
            "max_passage_length": 256,
            "batch_size": 32,
            "num_workers": 4
        },
        "storage": {
            "cache_dir": "~/.cache/kg_dense_research",
            "temp_dir": "/tmp/kg_dense_processing"
        }
    }
    
    config_path = base_path / "data_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created configuration: {config_path}")
    return config_path

def create_download_script(base_path):
    """Create master download script"""
    
    script_content = '''#!/bin/bash

# Master Dataset Download Script
# Downloads all datasets for KG + Dense Vector Research

set -e  # Exit on any error

echo "Starting dataset download process..."
echo "This will download approximately 50GB of data"
echo "Ensure you have sufficient disk space and internet bandwidth"
echo ""

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

# Initialize conda
if [ -f "$HOME/miniforge3/bin/conda" ]; then
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
fi

# Activate environment
conda activate kg-dense

# Create necessary directories
mkdir -p data/raw data/processed logs/preprocessing

# Download datasets in order of priority
echo "=== Phase 1: QA Datasets ==="
python scripts/data_processing/download_natural_questions.py
python scripts/data_processing/download_ms_marco_qa.py

echo "=== Phase 2: IR Datasets ==="
python scripts/data_processing/download_ms_marco_passage.py
python scripts/data_processing/download_beir.py

echo "=== Phase 3: KG Dataset ==="
python scripts/data_processing/download_wikidata5m.py

echo "=== Download Complete ==="
echo "All datasets downloaded successfully!"
echo "Next: Run preprocessing scripts"
'''
    
    script_path = base_path / "download_all_datasets.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    print(f"Created download script: {script_path}")
    return script_path

def main():
    """Main setup function"""
    print("Setting up data structure for KG + Dense Vector Research")
    print("=" * 60)
    
    # Get current project directory
    project_dir = Path.cwd()
    print(f"Project directory: {project_dir}")
    
    # Create directory structure
    print("\n1. Creating directory structure...")
    base_path = create_data_structure(project_dir)
    
    # Create configuration
    print("\n2. Creating dataset configuration...")
    config_path = create_dataset_config(project_dir)
    
    # Create download script
    print("\n3. Creating download script...")
    script_path = create_download_script(project_dir)
    
    print("\n" + "=" * 60)
    print("✅ Data structure setup complete!")
    print(f"📁 Base path: {base_path}")
    print(f"⚙️  Config: {config_path}")
    print(f"📜 Download script: {script_path}")
    print("\nNext steps:")
    print("1. Review data_config.json")
    print("2. Run individual dataset download scripts")
    print("3. Or run ./download_all_datasets.sh for everything")

if __name__ == "__main__":
    main()
