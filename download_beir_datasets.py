#!/usr/bin/env python3
"""
Download BEIR benchmark datasets using the official beir library
Alternative approach for BEIR dataset download
"""

import os
import logging
from pathlib import Path
import subprocess
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_beir():
    """Install beir library if not available"""
    try:
        import beir
        logger.info("BEIR library already installed")
        return True
    except ImportError:
        logger.info("Installing BEIR library...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "beir"])
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install BEIR: {e}")
            return False

def download_beir_datasets():
    """Download BEIR datasets using official library"""
    if not install_beir():
        return False
    
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    
    base_dir = Path.cwd()
    beir_dir = base_dir / "data" / "raw" / "ir" / "beir"
    beir_dir.mkdir(parents=True, exist_ok=True)
    
    # Small BEIR datasets for development
    datasets = ["nfcorpus", "scifact", "arguana", "fiqa"]
    
    for dataset_name in datasets:
        try:
            logger.info(f"Downloading BEIR {dataset_name}...")
            
            # Download dataset
            dataset_path = beir_dir / dataset_name
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            data_path = util.download_and_unzip(url, str(beir_dir))
            
            # Load and verify
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
            
            logger.info(f"✓ {dataset_name}: {len(queries)} queries, {len(corpus)} docs")
            
        except Exception as e:
            logger.error(f"✗ Failed to download {dataset_name}: {e}")
            continue
    
    logger.info("BEIR datasets download completed")
    return True

if __name__ == "__main__":
    download_beir_datasets()
