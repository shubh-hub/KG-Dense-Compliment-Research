#!/usr/bin/env python3
"""
Detailed Data Analysis for KG + Dense Vector Complementarity Research
Analyzes all downloaded datasets and prepares processing recommendations
Optimized for Apple M1 MacBook Air
"""

import os
import json
import gzip
import tarfile
import zipfile
from pathlib import Path
from collections import defaultdict
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetailedDataAnalyzer:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.data_dir = self.base_dir / "data"
        self.analysis = {
            "timestamp": datetime.now().isoformat(),
            "datasets": {},
            "processing_recommendations": {},
            "memory_estimates": {},
            "summary": {}
        }
    
    def get_file_size_mb(self, file_path):
        """Get file size in MB"""
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except:
            return 0
    
    def count_files_in_directory(self, directory):
        """Count files recursively in directory"""
        try:
            return sum(1 for _ in Path(directory).rglob('*') if _.is_file())
        except:
            return 0
    
    def analyze_wikidata5m_kg(self):
        """Analyze Wikidata5M KG dataset"""
        logger.info("Analyzing Wikidata5M KG dataset...")
        
        kg_dir = self.data_dir / "raw" / "kg" / "wikidata5m"
        analysis = {
            "task_type": "kg",
            "status": "downloaded",
            "files": {},
            "total_size_mb": 0,
            "processing_complexity": "medium"
        }
        
        if not kg_dir.exists():
            analysis["status"] = "missing"
            self.analysis["datasets"]["wikidata5m_kg"] = analysis
            return
        
        # Analyze each file
        key_files = [
            "entities.txt.gz",
            "relations.txt.gz", 
            "triples_train.txt.gz",
            "wikidata5m_text.txt.gz",
            "wikidata5m_alias.tar.gz",
            "wikidata5m_transductive.tar.gz",
            "wikidata5m_all_triplet.txt.gz"
        ]
        
        for file_name in key_files:
            file_path = kg_dir / file_name
            if file_path.exists():
                size_mb = self.get_file_size_mb(file_path)
                analysis["files"][file_name] = {
                    "size_mb": size_mb,
                    "exists": True,
                    "compressed": file_name.endswith(('.gz', '.tar.gz'))
                }
                analysis["total_size_mb"] += size_mb
            else:
                analysis["files"][file_name] = {
                    "size_mb": 0,
                    "exists": False,
                    "compressed": False
                }
        
        # Processing recommendations
        analysis["processing_recommendations"] = {
            "memory_limit_mb": 2048,  # M1 MacBook limit
            "batch_size": 1000,
            "streaming_required": True,
            "estimated_entities": 5000000,
            "estimated_relations": 822,
            "estimated_triples": 20000000
        }
        
        self.analysis["datasets"]["wikidata5m_kg"] = analysis
    
    def analyze_natural_questions_qa(self):
        """Analyze Natural Questions QA dataset"""
        logger.info("Analyzing Natural Questions QA dataset...")
        
        nq_dir = self.data_dir / "raw" / "qa" / "natural_questions"
        analysis = {
            "task_type": "qa",
            "status": "downloaded",
            "cache_structure": {},
            "total_size_mb": 0,
            "processing_complexity": "high"
        }
        
        if not nq_dir.exists():
            analysis["status"] = "missing"
            self.analysis["datasets"]["natural_questions_qa"] = analysis
            return
        
        # Analyze cache structure
        downloads_dir = nq_dir / "downloads"
        cache_dir = nq_dir / "cache"
        
        if downloads_dir.exists():
            download_files = list(downloads_dir.glob("*"))
            analysis["cache_structure"]["downloads"] = {
                "file_count": len([f for f in download_files if f.is_file()]),
                "total_size_mb": sum(self.get_file_size_mb(f) for f in download_files if f.is_file()),
                "sample_files": [f.name for f in download_files[:5] if f.is_file()]
            }
            analysis["total_size_mb"] += analysis["cache_structure"]["downloads"]["total_size_mb"]
        
        if cache_dir.exists():
            cache_files = list(cache_dir.rglob("*"))
            analysis["cache_structure"]["cache"] = {
                "file_count": len([f for f in cache_files if f.is_file()]),
                "total_size_mb": sum(self.get_file_size_mb(f) for f in cache_files if f.is_file())
            }
            analysis["total_size_mb"] += analysis["cache_structure"]["cache"]["total_size_mb"]
        
        # Processing recommendations
        analysis["processing_recommendations"] = {
            "memory_limit_mb": 4096,  # Higher for NQ processing
            "streaming_required": True,
            "batch_size": 100,
            "sample_size": 15000,  # For M1 compatibility
            "estimated_examples": 307373,
            "use_hf_streaming": True
        }
        
        self.analysis["datasets"]["natural_questions_qa"] = analysis
    
    def analyze_ms_marco_qa(self):
        """Analyze MS MARCO QA dataset"""
        logger.info("Analyzing MS MARCO QA dataset...")
        
        marco_dir = self.data_dir / "raw" / "qa" / "ms_marco_qa"
        analysis = {
            "task_type": "qa",
            "status": "downloaded",
            "splits": {},
            "total_size_mb": 0,
            "processing_complexity": "low"
        }
        
        if not marco_dir.exists():
            analysis["status"] = "missing"
            self.analysis["datasets"]["ms_marco_qa"] = analysis
            return
        
        # Analyze splits
        for split in ["train", "validation", "test"]:
            split_dir = marco_dir / split
            if split_dir.exists():
                files = list(split_dir.glob("*"))
                split_size = sum(self.get_file_size_mb(f) for f in files if f.is_file())
                analysis["splits"][split] = {
                    "file_count": len([f for f in files if f.is_file()]),
                    "size_mb": split_size,
                    "files": [f.name for f in files if f.is_file()]
                }
                analysis["total_size_mb"] += split_size
        
        # Processing recommendations
        analysis["processing_recommendations"] = {
            "memory_limit_mb": 1024,
            "streaming_required": False,
            "batch_size": 1000,
            "estimated_examples": 909000,
            "ready_for_processing": True
        }
        
        self.analysis["datasets"]["ms_marco_qa"] = analysis
    
    def analyze_beir_ir_datasets(self):
        """Analyze BEIR IR datasets"""
        logger.info("Analyzing BEIR IR datasets...")
        
        ir_dir = self.data_dir / "raw" / "ir"
        analysis = {
            "task_type": "ir",
            "status": "downloaded",
            "datasets": {},
            "total_size_mb": 0,
            "processing_complexity": "low"
        }
        
        if not ir_dir.exists():
            analysis["status"] = "missing"
            self.analysis["datasets"]["beir_ir"] = analysis
            return
        
        # Analyze individual BEIR datasets
        beir_datasets = ["arguana", "fiqa", "nfcorpus", "scifact"]
        
        for dataset in beir_datasets:
            dataset_dir = ir_dir / dataset
            zip_file = ir_dir / f"{dataset}.zip"
            
            dataset_info = {
                "directory_exists": dataset_dir.exists(),
                "zip_exists": zip_file.exists(),
                "size_mb": 0,
                "files": []
            }
            
            if dataset_dir.exists():
                files = list(dataset_dir.rglob("*"))
                dataset_info["size_mb"] += sum(self.get_file_size_mb(f) for f in files if f.is_file())
                dataset_info["files"] = [f.name for f in files if f.is_file()]
            
            if zip_file.exists():
                dataset_info["size_mb"] += self.get_file_size_mb(zip_file)
            
            analysis["datasets"][dataset] = dataset_info
            analysis["total_size_mb"] += dataset_info["size_mb"]
        
        # MS MARCO Passage
        ms_marco_passage_dir = ir_dir / "ms_marco_passage"
        if ms_marco_passage_dir.exists():
            files = list(ms_marco_passage_dir.rglob("*"))
            passage_size = sum(self.get_file_size_mb(f) for f in files if f.is_file())
            analysis["datasets"]["ms_marco_passage"] = {
                "directory_exists": True,
                "size_mb": passage_size,
                "files": [f.name for f in files if f.is_file()]
            }
            analysis["total_size_mb"] += passage_size
        
        # Processing recommendations
        analysis["processing_recommendations"] = {
            "memory_limit_mb": 512,
            "streaming_required": False,
            "batch_size": 1000,
            "ready_for_processing": True,
            "estimated_total_examples": 15000
        }
        
        self.analysis["datasets"]["beir_ir"] = analysis
    
    def calculate_memory_estimates(self):
        """Calculate memory estimates for processing"""
        logger.info("Calculating memory estimates...")
        
        estimates = {
            "wikidata5m_processing": {
                "peak_memory_mb": 3072,
                "streaming_memory_mb": 1024,
                "recommended_approach": "streaming"
            },
            "natural_questions_processing": {
                "peak_memory_mb": 8192,
                "streaming_memory_mb": 2048,
                "recommended_approach": "streaming_with_sampling"
            },
            "ms_marco_processing": {
                "peak_memory_mb": 1024,
                "streaming_memory_mb": 512,
                "recommended_approach": "batch_processing"
            },
            "beir_processing": {
                "peak_memory_mb": 512,
                "streaming_memory_mb": 256,
                "recommended_approach": "batch_processing"
            }
        }
        
        self.analysis["memory_estimates"] = estimates
    
    def generate_processing_recommendations(self):
        """Generate comprehensive processing recommendations"""
        logger.info("Generating processing recommendations...")
        
        recommendations = {
            "processing_order": [
                "beir_ir",
                "ms_marco_qa", 
                "wikidata5m_kg",
                "natural_questions_qa"
            ],
            "m1_optimizations": {
                "environment_variables": {
                    "KMP_DUPLICATE_LIB_OK": "TRUE",
                    "TOKENIZERS_PARALLELISM": "false",
                    "PYTORCH_ENABLE_MPS_FALLBACK": "1"
                },
                "memory_management": {
                    "max_memory_usage": "6GB",
                    "use_streaming": True,
                    "batch_sizes": {
                        "kg_processing": 1000,
                        "qa_processing": 100,
                        "ir_processing": 1000
                    }
                }
            },
            "data_preparation_steps": {
                "step_1": "Extract and validate Wikidata5M compressed files",
                "step_2": "Create streaming loaders for Natural Questions",
                "step_3": "Validate MS MARCO QA structure",
                "step_4": "Process BEIR datasets",
                "step_5": "Generate unified dataset analysis"
            },
            "expected_processing_time": {
                "wikidata5m": "10-15 minutes",
                "natural_questions": "15-20 minutes", 
                "ms_marco_qa": "2-3 minutes",
                "beir_ir": "2-3 minutes",
                "total": "30-45 minutes"
            }
        }
        
        self.analysis["processing_recommendations"] = recommendations
    
    def generate_summary(self):
        """Generate analysis summary"""
        logger.info("Generating summary...")
        
        total_datasets = len(self.analysis["datasets"])
        total_size = sum(
            dataset.get("total_size_mb", 0) 
            for dataset in self.analysis["datasets"].values()
        )
        
        ready_datasets = sum(
            1 for dataset in self.analysis["datasets"].values()
            if dataset.get("status") == "downloaded"
        )
        
        summary = {
            "total_datasets": total_datasets,
            "ready_datasets": ready_datasets,
            "total_size_gb": round(total_size / 1024, 2),
            "datasets_by_task": {
                "kg": 1,
                "qa": 2, 
                "ir": 1
            },
            "processing_readiness": {
                "ready_for_processing": ready_datasets == total_datasets,
                "estimated_processing_time": "30-45 minutes",
                "memory_requirements": "6GB peak, 2GB streaming"
            },
            "complementarity_research_status": "Ready to begin after processing"
        }
        
        self.analysis["summary"] = summary
    
    def run_analysis(self):
        """Run complete data analysis"""
        logger.info("Starting detailed data analysis...")
        
        # Analyze each dataset
        self.analyze_wikidata5m_kg()
        self.analyze_natural_questions_qa()
        self.analyze_ms_marco_qa()
        self.analyze_beir_ir_datasets()
        
        # Generate estimates and recommendations
        self.calculate_memory_estimates()
        self.generate_processing_recommendations()
        self.generate_summary()
        
        logger.info("Data analysis complete!")
        return self.analysis
    
    def save_analysis(self, output_file="detailed_data_analysis.json"):
        """Save analysis to file"""
        output_path = self.base_dir / "data" / "analysis" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.analysis, f, indent=2)
        
        logger.info(f"Analysis saved to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("DETAILED DATA ANALYSIS SUMMARY")
        print("="*60)
        
        summary = self.analysis["summary"]
        print(f"Total Datasets: {summary['total_datasets']}")
        print(f"Ready Datasets: {summary['ready_datasets']}")
        print(f"Total Size: {summary['total_size_gb']} GB")
        print(f"Processing Time: {summary['processing_readiness']['estimated_processing_time']}")
        print(f"Memory Required: {summary['processing_readiness']['memory_requirements']}")
        
        print("\nDATASET BREAKDOWN:")
        for name, dataset in self.analysis["datasets"].items():
            status = "✅" if dataset.get("status") == "downloaded" else "❌"
            size = dataset.get("total_size_mb", 0)
            print(f"{status} {name}: {size:.1f} MB ({dataset.get('task_type', 'unknown')})")
        
        print("\nPROCESSING RECOMMENDATIONS:")
        recs = self.analysis["processing_recommendations"]
        print(f"Processing Order: {' → '.join(recs['processing_order'])}")
        print(f"M1 Optimizations: Environment variables and streaming enabled")
        print(f"Expected Total Time: {recs['expected_processing_time']['total']}")
        
        print("\n" + "="*60)
        if summary["processing_readiness"]["ready_for_processing"]:
            print("🎉 ALL DATASETS READY FOR PROCESSING!")
            print("Next step: Run the processing pipeline")
        else:
            print("⚠️  Some datasets need attention before processing")
        print("="*60)

def main():
    """Main execution function"""
    analyzer = DetailedDataAnalyzer()
    
    # Run analysis
    analysis = analyzer.run_analysis()
    
    # Save results
    output_file = analyzer.save_analysis()
    
    # Print summary
    analyzer.print_summary()
    
    return analysis, output_file

if __name__ == "__main__":
    main()
