#!/usr/bin/env python3
"""
Download remaining datasets for KG + Dense Vector research
Optimized for M1 MacBook with 50GB available space

Downloads in order of size (smallest first):
1. BEIR datasets (~0.15GB)
2. Wikidata5M KG (1.8GB) 
3. MS MARCO QA (2.5GB)
4. MS MARCO Passage (3.2GB)

Total: ~7.65GB additional
"""

import os
import json
import logging
from pathlib import Path
from datasets import load_dataset
import requests
import tarfile
import zipfile
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dataset_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        
        # Create directories
        for subdir in ["qa", "ir", "kg"]:
            (self.raw_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        # Load config
        with open(self.base_dir / "data_config.json", 'r') as f:
            self.config = json.load(f)
    
    def check_disk_space(self):
        """Check available disk space"""
        import shutil
        total, used, free = shutil.disk_usage(self.base_dir)
        free_gb = free / (1024**3)
        logger.info(f"Available disk space: {free_gb:.1f} GB")
        return free_gb
    
    def download_beir_datasets(self):
        """Download BEIR benchmark datasets (smallest first)"""
        logger.info("Starting BEIR datasets download...")
        beir_dir = self.raw_dir / "ir" / "beir"
        beir_dir.mkdir(parents=True, exist_ok=True)
        
        beir_datasets = self.config["datasets"]["ir"]["beir"]["datasets"]
        
        for dataset_name, info in beir_datasets.items():
            logger.info(f"Downloading BEIR {dataset_name} ({info['size_gb']} GB)...")
            
            try:
                # Use datasets library for BEIR
                dataset = load_dataset("BeIR/beir", dataset_name)
                
                # Save to disk
                dataset_dir = beir_dir / dataset_name
                dataset.save_to_disk(str(dataset_dir))
                
                logger.info(f"✓ BEIR {dataset_name} downloaded successfully")
                
            except Exception as e:
                logger.error(f"✗ Failed to download BEIR {dataset_name}: {e}")
                continue
        
        logger.info("BEIR datasets download completed")
    
    def download_wikidata5m(self):
        """Download Wikidata5M knowledge graph"""
        logger.info("Starting Wikidata5M download (1.8GB)...")
        kg_dir = self.raw_dir / "kg" / "wikidata5m"
        kg_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download from official source
            urls = {
                "entities": "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_entity.txt.gz",
                "relations": "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_relation.txt.gz", 
                "triples_train": "https://www.dropbox.com/s/563omb11cxaqr83/wikidata5m_inductive_train.txt.gz",
                "triples_valid": "https://www.dropbox.com/s/10ix4lz5duqx8kg/wikidata5m_inductive_valid.txt.gz",
                "triples_test": "https://www.dropbox.com/s/s8csmi9gcb1a1ug/wikidata5m_inductive_test.txt.gz"
            }
            
            for name, url in urls.items():
                file_path = kg_dir / f"{name}.txt.gz"
                if file_path.exists():
                    logger.info(f"✓ {name} already exists, skipping")
                    continue
                    
                logger.info(f"Downloading {name}...")
                self._download_file(url, file_path)
                
            logger.info("✓ Wikidata5M download completed")
            
        except Exception as e:
            logger.error(f"✗ Failed to download Wikidata5M: {e}")
    
    def download_ms_marco_qa(self):
        """Download MS MARCO QA dataset"""
        logger.info("Starting MS MARCO QA download (2.5GB)...")
        qa_dir = self.raw_dir / "qa" / "ms_marco_qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use datasets library
            dataset = load_dataset("microsoft/ms_marco", "v2.1")
            dataset.save_to_disk(str(qa_dir))
            
            logger.info("✓ MS MARCO QA download completed")
            
        except Exception as e:
            logger.error(f"✗ Failed to download MS MARCO QA: {e}")
    
    def download_ms_marco_passage(self):
        """Download MS MARCO Passage dataset"""
        logger.info("Starting MS MARCO Passage download (3.2GB)...")
        ir_dir = self.raw_dir / "ir" / "ms_marco_passage"
        ir_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download passage ranking dataset
            dataset = load_dataset("microsoft/ms_marco", "v1.1")
            dataset.save_to_disk(str(ir_dir))
            
            logger.info("✓ MS MARCO Passage download completed")
            
        except Exception as e:
            logger.error(f"✗ Failed to download MS MARCO Passage: {e}")
    
    def _download_file(self, url, file_path):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f, tqdm(
            desc=file_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def create_download_summary(self):
        """Create summary of downloaded datasets"""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "datasets": {},
            "total_size_gb": 0
        }
        
        # Check each dataset directory
        for task in ["qa", "ir", "kg"]:
            task_dir = self.raw_dir / task
            if task_dir.exists():
                for dataset_dir in task_dir.iterdir():
                    if dataset_dir.is_dir():
                        size_bytes = sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file())
                        size_gb = size_bytes / (1024**3)
                        summary["datasets"][dataset_dir.name] = {
                            "size_gb": round(size_gb, 2),
                            "path": str(dataset_dir)
                        }
                        summary["total_size_gb"] += size_gb
        
        summary["total_size_gb"] = round(summary["total_size_gb"], 2)
        
        # Save summary
        summary_file = self.data_dir / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    """Main download function"""
    downloader = DatasetDownloader()
    
    # Check disk space
    free_space = downloader.check_disk_space()
    if free_space < 10:  # Need at least 10GB free
        logger.error(f"Insufficient disk space: {free_space:.1f}GB available, need ~8GB + buffer")
        return
    
    logger.info("Starting dataset downloads in order of size...")
    
    # Download in order of size (smallest first)
    try:
        # 1. BEIR datasets (~0.15GB)
        downloader.download_beir_datasets()
        
        # 2. Wikidata5M (1.8GB)
        downloader.download_wikidata5m()
        
        # 3. MS MARCO QA (2.5GB)  
        downloader.download_ms_marco_qa()
        
        # 4. MS MARCO Passage (3.2GB)
        downloader.download_ms_marco_passage()
        
        # Create summary
        summary = downloader.create_download_summary()
        
        print("\n" + "="*60)
        print("DATASET DOWNLOAD COMPLETED!")
        print("="*60)
        print(f"Total downloaded: {summary['total_size_gb']:.1f} GB")
        print("\nDatasets ready:")
        for name, info in summary["datasets"].items():
            print(f"  • {name}: {info['size_gb']:.1f} GB")
        
        print(f"\nRemaining disk space: {downloader.check_disk_space():.1f} GB")
        print("\nNext steps:")
        print("1. Process existing Natural Questions data")
        print("2. Create evaluation frameworks")
        print("3. Begin model development")
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Download failed: {e}")

if __name__ == "__main__":
    main()
