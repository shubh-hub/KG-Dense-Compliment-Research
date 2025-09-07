#!/usr/bin/env python3
"""
Download Progress Checker
Monitors dataset download progress and provides status updates
"""

import os
import json
from pathlib import Path
import time

def check_directory_size(path):
    """Calculate directory size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, FileNotFoundError):
                pass
    return total_size / (1024 * 1024)  # Convert to MB

def check_dataset_progress():
    """Check progress of dataset downloads"""
    base_dir = Path.cwd()
    
    datasets = {
        "Natural Questions": {
            "path": base_dir / "data" / "raw" / "qa" / "natural_questions",
            "expected_size_gb": 42.0,
            "status": "downloading"
        },
        "MS MARCO QA": {
            "path": base_dir / "data" / "raw" / "qa" / "ms_marco_qa", 
            "expected_size_gb": 2.5,
            "status": "pending"
        }
    }
    
    print("Dataset Download Progress")
    print("=" * 50)
    
    for name, info in datasets.items():
        path = info["path"]
        expected_gb = info["expected_size_gb"]
        
        if path.exists():
            current_mb = check_directory_size(path)
            current_gb = current_mb / 1024
            progress = (current_gb / expected_gb) * 100
            
            print(f"\n{name}:")
            print(f"  Path: {path}")
            print(f"  Current size: {current_gb:.2f} GB")
            print(f"  Expected size: {expected_gb} GB")
            print(f"  Progress: {progress:.1f}%")
            
            if progress >= 95:
                print(f"  Status: ✅ Complete")
            elif current_gb > 0:
                print(f"  Status: 🔄 Downloading...")
            else:
                print(f"  Status: ⏳ Starting...")
        else:
            print(f"\n{name}:")
            print(f"  Status: ❌ Not started")
    
    # Check processed data
    print(f"\nProcessed Data:")
    processed_base = base_dir / "data" / "processed" / "qa"
    
    for dataset_dir in ["natural_questions", "ms_marco_qa"]:
        processed_path = processed_base / dataset_dir
        if processed_path.exists():
            files = list(processed_path.glob("*.jsonl"))
            if files:
                print(f"  {dataset_dir}: ✅ {len(files)} processed files")
            else:
                print(f"  {dataset_dir}: ⏳ Processing...")
        else:
            print(f"  {dataset_dir}: ❌ Not processed")

if __name__ == "__main__":
    check_dataset_progress()
