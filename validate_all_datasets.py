#!/usr/bin/env python3
"""
Comprehensive Dataset Validation Script
Validates all processed datasets for complementarity research
"""

import os
import json
import glob
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetValidator:
    """Comprehensive validation of all processed datasets"""
    
    def __init__(self, base_dir: str = "/Users/shivam/Documents/Shubham/Research project"):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.processed_dir = self.data_dir / "processed"
        
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {},
            'summary': {},
            'errors': []
        }
    
    def validate_beir_ir(self) -> Dict[str, Any]:
        """Validate BEIR IR datasets"""
        logger.info("Validating BEIR IR datasets...")
        
        ir_dir = self.processed_dir / "ir"
        results = {
            'name': 'BEIR_IR',
            'path': str(ir_dir),
            'exists': ir_dir.exists(),
            'datasets': {},
            'total_examples': 0,
            'status': 'unknown'
        }
        
        if not ir_dir.exists():
            results['status'] = 'missing'
            return results
        
        # Check for BEIR JSONL files (correct format from processing)
        beir_files = list(ir_dir.glob("beir_*.jsonl"))
        
        for file_path in beir_files:
            dataset_name = file_path.stem.replace('beir_', '')
            try:
                with open(file_path, 'r') as f:
                    examples = [json.loads(line) for line in f if line.strip()]
                
                results['datasets'][dataset_name] = {
                    'file': str(file_path),
                    'examples': len(examples),
                    'exists': True
                }
                results['total_examples'] += len(examples)
            except Exception as e:
                results['datasets'][dataset_name] = {
                    'file': str(file_path),
                    'error': str(e),
                    'exists': True
                }
        
        results['status'] = 'valid' if results['total_examples'] > 0 else 'invalid'
        return results
    
    def validate_ms_marco_qa(self) -> Dict[str, Any]:
        """Validate MS MARCO QA dataset"""
        logger.info("Validating MS MARCO QA dataset...")
        
        qa_dir = self.processed_dir / "qa" / "ms_marco_qa"
        results = {
            'name': 'MS_MARCO_QA',
            'path': str(qa_dir),
            'exists': qa_dir.exists(),
            'splits': {},
            'total_examples': 0,
            'status': 'unknown'
        }
        
        if not qa_dir.exists():
            results['status'] = 'missing'
            return results
        
        # Check for MS MARCO QA JSONL files in subdirectory
        splits = ['train', 'validation', 'test']
        
        for split in splits:
            file_path = qa_dir / f"{split}.jsonl"
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        examples = [json.loads(line) for line in f if line.strip()]
                    
                    results['splits'][split] = {
                        'file': str(file_path),
                        'examples': len(examples),
                        'sample': examples[0] if examples else None
                    }
                    results['total_examples'] += len(examples)
                    
                except Exception as e:
                    results['splits'][split] = {
                        'file': str(file_path),
                        'error': str(e)
                    }
            else:
                results['splits'][split] = {
                    'file': str(file_path),
                    'exists': False
                }
        
        results['status'] = 'valid' if results['total_examples'] > 0 else 'invalid'
        return results
    
    def validate_natural_questions(self) -> Dict[str, Any]:
        """Validate Natural Questions dataset"""
        logger.info("Validating Natural Questions dataset...")
        
        qa_dir = self.processed_dir / "qa"
        results = {
            'name': 'Natural_Questions',
            'path': str(qa_dir),
            'exists': qa_dir.exists(),
            'batches': {},
            'total_examples': 0,
            'status': 'unknown'
        }
        
        if not qa_dir.exists():
            results['status'] = 'missing'
            return results
        
        # Check for Natural Questions batch files
        nq_files = list(qa_dir.glob("natural_questions_batch_*.json"))
        
        for file_path in nq_files:
            batch_name = file_path.stem
            try:
                with open(file_path, 'r') as f:
                    examples = [json.loads(line) for line in f if line.strip()]
                
                results['batches'][batch_name] = {
                    'file': str(file_path),
                    'examples': len(examples),
                    'sample': examples[0] if examples else None
                }
                results['total_examples'] += len(examples)
                
            except Exception as e:
                results['batches'][batch_name] = {
                    'file': str(file_path),
                    'error': str(e)
                }
        
        # Check for summary file
        summary_file = qa_dir / "natural_questions_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                results['summary'] = summary
            except Exception as e:
                results['summary_error'] = str(e)
        
        results['status'] = 'valid' if results['total_examples'] > 0 else 'processing'
        return results
    
    def validate_wikidata5m_kg(self) -> Dict[str, Any]:
        """Validate Wikidata5M KG dataset"""
        logger.info("Validating Wikidata5M KG dataset...")
        
        kg_dir = self.processed_dir / "kg"
        
        components = ['entities', 'relations', 'triples']
        total_examples = 0
        component_counts = {}
        
        for component in components:
            component_file = kg_dir / f"{component}.json"
            if component_file.exists():
                try:
                    # Try JSON array format first
                    with open(component_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # Filter out corrupted entries with binary data
                            valid_examples = []
                            for item in data:
                                if isinstance(item, dict):
                                    # Check if any values contain null bytes (binary corruption)
                                    has_corruption = False
                                    for key, value in item.items():
                                        if isinstance(value, str) and '\x00' in value:
                                            has_corruption = True
                                            break
                                    if not has_corruption:
                                        valid_examples.append(item)
                            
                            component_counts[component] = len(valid_examples)
                            total_examples += len(valid_examples)
                        else:
                            component_counts[component] = 0
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Try JSONL format
                    try:
                        with open(component_file, 'r') as f:
                            examples = []
                            for line in f:
                                if line.strip():
                                    try:
                                        item = json.loads(line)
                                        if isinstance(item, dict):
                                            # Check for binary corruption
                                            has_corruption = False
                                            for key, value in item.items():
                                                if isinstance(value, str) and '\x00' in value:
                                                    has_corruption = True
                                                    break
                                            if not has_corruption:
                                                examples.append(item)
                                    except json.JSONDecodeError:
                                        continue
                            component_counts[component] = len(examples)
                            total_examples += len(examples)
                    except Exception:
                        component_counts[component] = 0
            else:
                return {
                    'valid': False,
                    'examples': 0,
                    'error': f'Missing {component}.json file'
                }
        
        results = {
            'name': 'Wikidata5M_KG',
            'path': str(kg_dir),
            'exists': kg_dir.exists(),
            'total_examples': total_examples,
            'components': component_counts,
            'files': {
                'entities': str(kg_dir / "entities.json"),
                'relations': str(kg_dir / "relations.json"), 
                'triples': str(kg_dir / "triples.json")
            }
        }
        
        results['status'] = 'valid' if total_examples > 0 else 'invalid'
        return results
    
    def check_data_loaders(self) -> Dict[str, Any]:
        """Check if data loaders exist and are functional"""
        logger.info("Checking data loaders...")
        
        loaders_file = self.base_dir / "data_processing" / "data_loaders.py"
        results = {
            'name': 'Data_Loaders',
            'file': str(loaders_file),
            'exists': loaders_file.exists(),
            'status': 'valid' if loaders_file.exists() else 'missing'
        }
        
        return results
    
    def check_model_architectures(self) -> Dict[str, Any]:
        """Check if model architectures exist"""
        logger.info("Checking model architectures...")
        
        models_file = self.base_dir / "models" / "baseline_architectures.py"
        results = {
            'name': 'Model_Architectures',
            'file': str(models_file),
            'exists': models_file.exists(),
            'status': 'valid' if models_file.exists() else 'missing'
        }
        
        return results
    
    def check_evaluation_framework(self) -> Dict[str, Any]:
        """Check if evaluation framework exists"""
        logger.info("Checking evaluation framework...")
        
        eval_file = self.base_dir / "evaluation" / "evaluation_framework.py"
        results = {
            'name': 'Evaluation_Framework',
            'file': str(eval_file),
            'exists': eval_file.exists(),
            'status': 'valid' if eval_file.exists() else 'missing'
        }
        
        return results
    
    def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of all components"""
        logger.info("Starting comprehensive dataset validation...")
        
        # Validate datasets
        self.validation_results['datasets']['beir_ir'] = self.validate_beir_ir()
        self.validation_results['datasets']['ms_marco_qa'] = self.validate_ms_marco_qa()
        self.validation_results['datasets']['natural_questions'] = self.validate_natural_questions()
        self.validation_results['datasets']['wikidata5m_kg'] = self.validate_wikidata5m_kg()
        
        # Check infrastructure
        self.validation_results['infrastructure'] = {
            'data_loaders': self.check_data_loaders(),
            'model_architectures': self.check_model_architectures(),
            'evaluation_framework': self.check_evaluation_framework()
        }
        
        # Generate summary
        self.generate_summary()
        
        # Save validation report
        self.save_validation_report()
        
        return self.validation_results
    
    def generate_summary(self):
        """Generate validation summary"""
        summary = {
            'total_datasets': len(self.validation_results['datasets']),
            'valid_datasets': 0,
            'total_examples': 0,
            'ready_for_training': False
        }
        
        for dataset_name, dataset_info in self.validation_results['datasets'].items():
            if dataset_info['status'] == 'valid':
                summary['valid_datasets'] += 1
                summary['total_examples'] += dataset_info.get('total_examples', 0)
        
        # Check if ready for training
        required_components = ['beir_ir', 'ms_marco_qa', 'wikidata5m_kg']
        valid_components = [name for name in required_components 
                          if self.validation_results['datasets'][name]['status'] == 'valid']
        
        summary['ready_for_training'] = len(valid_components) >= 3
        summary['missing_components'] = [name for name in required_components 
                                       if name not in valid_components]
        
        self.validation_results['summary'] = summary
    
    def save_validation_report(self):
        """Save validation report to file"""
        report_file = self.base_dir / "data" / "analysis" / "validation_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {report_file}")
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("DATASET VALIDATION SUMMARY")
        print("="*60)
        
        summary = self.validation_results['summary']
        
        print(f"Total Datasets: {summary['total_datasets']}")
        print(f"Valid Datasets: {summary['valid_datasets']}")
        print(f"Total Examples: {summary['total_examples']:,}")
        print(f"Ready for Training: {'✓' if summary['ready_for_training'] else '✗'}")
        
        if summary.get('missing_components'):
            print(f"Missing Components: {', '.join(summary['missing_components'])}")
        
        print("\nDETAILED RESULTS:")
        print("-" * 40)
        
        for dataset_name, dataset_info in self.validation_results['datasets'].items():
            status_icon = "✓" if dataset_info['status'] == 'valid' else "✗" if dataset_info['status'] == 'invalid' else "⏳"
            examples = dataset_info.get('total_examples', 0)
            print(f"{status_icon} {dataset_name}: {examples:,} examples ({dataset_info['status']})")
        
        print("\nINFRASTRUCTURE:")
        print("-" * 40)
        
        for component_name, component_info in self.validation_results['infrastructure'].items():
            status_icon = "✓" if component_info['status'] == 'valid' else "✗"
            print(f"{status_icon} {component_name}: {component_info['status']}")

def main():
    """Main validation function"""
    validator = DatasetValidator()
    results = validator.run_validation()
    validator.print_summary()
    
    return results

if __name__ == "__main__":
    main()
