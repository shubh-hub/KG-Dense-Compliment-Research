#!/usr/bin/env python3
"""
Comprehensive Dataset Fixing Script for KG + Dense Vector Research
Fixes Wikidata5M, processes Natural Questions, and downloads BEIR datasets
"""

import os
import sys
import json
import gzip
import logging
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'dataset_fixing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveDatasetFixer:
    """Fix all missing datasets for KG + Dense Vector research"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.analysis_dir = self.data_dir / "analysis"
        
        # Ensure directories exist
        for dir_path in [self.raw_dir, self.processed_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Dataset configurations
        self.wikidata5m_urls = [
            "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_entity.txt.gz?dl=1",
            "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_relation.txt.gz?dl=1", 
            "https://www.dropbox.com/s/563omb11cxaqr83/wikidata5m_transductive_train.txt.gz?dl=1",
            "https://www.dropbox.com/s/5g0yn065dimh5us/wikidata5m_transductive_valid.txt.gz?dl=1",
            "https://www.dropbox.com/s/aeyn2a1qb9btdw3/wikidata5m_transductive_test.txt.gz?dl=1"
        ]
        
        self.beir_datasets = ["nfcorpus", "scifact", "arguana", "fiqa"]
        
    def download_file(self, url: str, filepath: Path, max_retries: int = 3) -> bool:
        """Download file with retry logic"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {url} to {filepath} (attempt {attempt + 1})")
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify file is not HTML
                with open(filepath, 'rb') as f:
                    first_bytes = f.read(100)
                    if b'<html' in first_bytes.lower() or b'<!doctype' in first_bytes.lower():
                        logger.error(f"Downloaded file is HTML, not data: {filepath}")
                        filepath.unlink()
                        continue
                
                logger.info(f"Successfully downloaded {filepath} ({filepath.stat().st_size} bytes)")
                return True
                
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if filepath.exists():
                    filepath.unlink()
                time.sleep(5)
        
        return False
    
    def fix_wikidata5m(self) -> bool:
        """Fix Wikidata5M KG dataset by re-downloading from reliable sources"""
        logger.info("=== PHASE 1: Fixing Wikidata5M KG Dataset ===")
        
        kg_dir = self.raw_dir / "kg" / "wikidata5m"
        kg_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove corrupted files
        for file in kg_dir.glob("*.gz"):
            logger.info(f"Removing corrupted file: {file}")
            file.unlink()
        
        # Alternative download URLs for Wikidata5M
        file_mappings = [
            ("https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_entity.txt.gz?dl=1", "entities.txt.gz"),
            ("https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_relation.txt.gz?dl=1", "relations.txt.gz"),
            ("https://www.dropbox.com/s/563omb11cxaqr83/wikidata5m_transductive_train.txt.gz?dl=1", "triples_train.txt.gz"),
            ("https://www.dropbox.com/s/5g0yn065dimh5us/wikidata5m_transductive_valid.txt.gz?dl=1", "triples_valid.txt.gz"),
            ("https://www.dropbox.com/s/aeyn2a1qb9btdw3/wikidata5m_transductive_test.txt.gz?dl=1", "triples_test.txt.gz")
        ]
        
        # Try alternative sources if Dropbox fails
        alternative_sources = [
            "https://github.com/deepmind/kg_geometry/raw/master/wikidata5m/",
            "https://huggingface.co/datasets/wikidata5m/resolve/main/"
        ]
        
        success_count = 0
        for url, filename in file_mappings:
            filepath = kg_dir / filename
            
            # Try primary URL
            if self.download_file(url, filepath):
                success_count += 1
                continue
            
            # Try alternative sources
            for alt_base in alternative_sources:
                alt_url = alt_base + filename.replace('.txt.gz', '.txt.gz')
                if self.download_file(alt_url, filepath):
                    success_count += 1
                    break
            else:
                logger.error(f"Failed to download {filename} from all sources")
        
        if success_count >= 3:  # Need at least entities, relations, and train triples
            logger.info(f"Successfully downloaded {success_count}/5 Wikidata5M files")
            return self.process_wikidata5m()
        else:
            logger.error("Failed to download sufficient Wikidata5M files")
            return False
    
    def process_wikidata5m(self) -> bool:
        """Process Wikidata5M files and create research-ready format"""
        try:
            kg_dir = self.raw_dir / "kg" / "wikidata5m"
            processed_kg_dir = self.processed_dir / "kg"
            processed_kg_dir.mkdir(parents=True, exist_ok=True)
            
            # Process entities
            entities = {}
            entities_file = kg_dir / "entities.txt.gz"
            if entities_file.exists():
                with gzip.open(entities_file, 'rt', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 100000:  # Limit for M1 processing
                            break
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            entities[parts[0]] = parts[1]
                logger.info(f"Processed {len(entities)} entities")
            
            # Process relations
            relations = {}
            relations_file = kg_dir / "relations.txt.gz"
            if relations_file.exists():
                with gzip.open(relations_file, 'rt', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            relations[parts[0]] = parts[1]
                logger.info(f"Processed {len(relations)} relations")
            
            # Process triples
            all_triples = []
            for split in ['train', 'valid', 'test']:
                triples_file = kg_dir / f"triples_{split}.txt.gz"
                if triples_file.exists():
                    split_triples = []
                    with gzip.open(triples_file, 'rt', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if split == 'train' and i >= 500000:  # Limit train set for M1
                                break
                            elif split in ['valid', 'test'] and i >= 50000:  # Limit dev/test
                                break
                            
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                triple = {
                                    'head': parts[0],
                                    'relation': parts[1], 
                                    'tail': parts[2],
                                    'head_label': entities.get(parts[0], parts[0]),
                                    'relation_label': relations.get(parts[1], parts[1]),
                                    'tail_label': entities.get(parts[2], parts[2])
                                }
                                split_triples.append(triple)
                                all_triples.append(triple)
                    
                    # Save split
                    split_file = processed_kg_dir / f"wikidata5m_{split}.jsonl"
                    with open(split_file, 'w') as f:
                        for triple in split_triples:
                            f.write(json.dumps(triple) + '\n')
                    logger.info(f"Saved {len(split_triples)} {split} triples to {split_file}")
            
            # Save metadata
            metadata = {
                'dataset': 'wikidata5m',
                'task_type': 'kg',
                'num_entities': len(entities),
                'num_relations': len(relations),
                'num_triples': len(all_triples),
                'splits': {
                    'train': len([t for t in all_triples if 'train' in str(t)]),
                    'valid': len([t for t in all_triples if 'valid' in str(t)]),
                    'test': len([t for t in all_triples if 'test' in str(t)])
                },
                'processed_at': datetime.now().isoformat()
            }
            
            with open(processed_kg_dir / "wikidata5m_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully processed Wikidata5M: {len(entities)} entities, {len(relations)} relations, {len(all_triples)} triples")
            return True
            
        except Exception as e:
            logger.error(f"Error processing Wikidata5M: {e}")
            return False
    
    def process_natural_questions(self) -> bool:
        """Process existing 18GB Natural Questions data from HuggingFace cache"""
        logger.info("=== PHASE 2: Processing Natural Questions Dataset ===")
        
        try:
            # Import datasets library
            from datasets import load_dataset
            
            # Load Natural Questions dataset
            logger.info("Loading Natural Questions from HuggingFace...")
            dataset = load_dataset("natural_questions", cache_dir=str(self.raw_dir / "qa" / "natural_questions"))
            
            processed_qa_dir = self.processed_dir / "qa"
            processed_qa_dir.mkdir(parents=True, exist_ok=True)
            
            total_processed = 0
            
            for split_name, split_data in dataset.items():
                logger.info(f"Processing {split_name} split with {len(split_data)} examples")
                
                processed_examples = []
                for i, example in enumerate(split_data):
                    if split_name == 'train' and i >= 100000:  # Limit for M1
                        break
                    elif split_name == 'validation' and i >= 10000:
                        break
                    
                    # Extract question and answers
                    question = example.get('question', {}).get('text', '')
                    if not question:
                        continue
                    
                    # Extract short answers
                    annotations = example.get('annotations', [])
                    short_answers = []
                    long_answers = []
                    
                    for annotation in annotations:
                        if annotation.get('short_answers'):
                            for short_ans in annotation['short_answers']:
                                if 'text' in short_ans:
                                    short_answers.append(short_ans['text'])
                        
                        if annotation.get('long_answer') and 'candidate_text' in annotation['long_answer']:
                            long_answers.append(annotation['long_answer']['candidate_text'])
                    
                    if short_answers or long_answers:
                        processed_example = {
                            'question': question,
                            'short_answers': short_answers,
                            'long_answers': long_answers,
                            'example_id': example.get('id', f"{split_name}_{i}"),
                            'split': split_name
                        }
                        processed_examples.append(processed_example)
                
                # Save processed split
                split_file = processed_qa_dir / f"natural_questions_{split_name}.jsonl"
                with open(split_file, 'w') as f:
                    for example in processed_examples:
                        f.write(json.dumps(example) + '\n')
                
                logger.info(f"Saved {len(processed_examples)} {split_name} examples to {split_file}")
                total_processed += len(processed_examples)
            
            # Save metadata
            metadata = {
                'dataset': 'natural_questions',
                'task_type': 'qa',
                'total_examples': total_processed,
                'processed_at': datetime.now().isoformat(),
                'source': 'huggingface_cache'
            }
            
            with open(processed_qa_dir / "natural_questions_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully processed Natural Questions: {total_processed} examples")
            return True
            
        except Exception as e:
            logger.error(f"Error processing Natural Questions: {e}")
            return False
    
    def fix_beir_datasets(self) -> bool:
        """Download and process BEIR IR benchmark datasets"""
        logger.info("=== PHASE 3: Fixing BEIR IR Datasets ===")
        
        try:
            # Install BEIR if not available
            try:
                import beir
            except ImportError:
                logger.info("Installing BEIR library...")
                subprocess.run([sys.executable, "-m", "pip", "install", "beir"], check=True)
                import beir
            
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader
            
            processed_ir_dir = self.processed_dir / "ir"
            processed_ir_dir.mkdir(parents=True, exist_ok=True)
            
            total_datasets = 0
            
            for dataset_name in self.beir_datasets:
                try:
                    logger.info(f"Processing BEIR dataset: {dataset_name}")
                    
                    # Download dataset
                    dataset_path = util.download_and_unzip(
                        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
                        str(self.raw_dir / "ir")
                    )
                    
                    # Load dataset
                    data_loader = GenericDataLoader(data_folder=dataset_path)
                    corpus, queries, qrels = data_loader.load(split="test")
                    
                    # Process and save
                    processed_examples = []
                    for query_id, query_text in queries.items():
                        if query_id in qrels:
                            for doc_id, relevance in qrels[query_id].items():
                                if doc_id in corpus:
                                    example = {
                                        'query': query_text,
                                        'query_id': query_id,
                                        'passage': corpus[doc_id].get('text', ''),
                                        'passage_id': doc_id,
                                        'relevance': relevance,
                                        'dataset': dataset_name
                                    }
                                    processed_examples.append(example)
                    
                    # Limit for M1 processing
                    if len(processed_examples) > 10000:
                        processed_examples = processed_examples[:10000]
                    
                    # Save dataset
                    dataset_file = processed_ir_dir / f"beir_{dataset_name}.jsonl"
                    with open(dataset_file, 'w') as f:
                        for example in processed_examples:
                            f.write(json.dumps(example) + '\n')
                    
                    logger.info(f"Saved {len(processed_examples)} examples for {dataset_name}")
                    total_datasets += 1
                    
                except Exception as e:
                    logger.error(f"Error processing BEIR dataset {dataset_name}: {e}")
                    continue
            
            # Save metadata
            metadata = {
                'datasets': self.beir_datasets,
                'task_type': 'ir',
                'processed_datasets': total_datasets,
                'processed_at': datetime.now().isoformat()
            }
            
            with open(processed_ir_dir / "beir_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully processed {total_datasets}/{len(self.beir_datasets)} BEIR datasets")
            return total_datasets > 0
            
        except Exception as e:
            logger.error(f"Error processing BEIR datasets: {e}")
            return False
    
    def validate_and_analyze(self) -> bool:
        """Validate all datasets and generate comprehensive analysis"""
        logger.info("=== PHASE 4: Validation & Analysis ===")
        
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'datasets': {},
                'summary': {}
            }
            
            total_examples = 0
            total_size_mb = 0
            datasets_by_task = {}
            
            # Analyze processed datasets
            for task_dir in self.processed_dir.iterdir():
                if task_dir.is_dir():
                    task_type = task_dir.name
                    
                    for dataset_file in task_dir.glob("*.jsonl"):
                        dataset_name = dataset_file.stem
                        
                        # Count examples and size
                        num_examples = 0
                        with open(dataset_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    num_examples += 1
                        
                        file_size_mb = dataset_file.stat().st_size / (1024 * 1024)
                        
                        # Sample data
                        sample_data = []
                        with open(dataset_file, 'r') as f:
                            for i, line in enumerate(f):
                                if i >= 3:  # First 3 examples
                                    break
                                if line.strip():
                                    sample_data.append(json.loads(line))
                        
                        analysis['datasets'][dataset_name] = {
                            'task_type': task_type,
                            'num_examples': num_examples,
                            'file_size_mb': file_size_mb,
                            'sample_data': sample_data
                        }
                        
                        total_examples += num_examples
                        total_size_mb += file_size_mb
                        
                        if task_type not in datasets_by_task:
                            datasets_by_task[task_type] = {'count': 0, 'examples': 0, 'size_mb': 0}
                        datasets_by_task[task_type]['count'] += 1
                        datasets_by_task[task_type]['examples'] += num_examples
                        datasets_by_task[task_type]['size_mb'] += file_size_mb
            
            analysis['summary'] = {
                'total_datasets': len(analysis['datasets']),
                'total_examples': total_examples,
                'total_size_mb': total_size_mb,
                'datasets_by_task': datasets_by_task
            }
            
            # Save analysis
            analysis_file = self.analysis_dir / "comprehensive_dataset_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Generate human-readable report
            report_lines = [
                "=" * 60,
                "COMPREHENSIVE DATASET ANALYSIS REPORT",
                "=" * 60,
                f"Generated: {analysis['timestamp']}",
                "",
                "DATASET SUMMARY:",
                f"  Total Datasets: {analysis['summary']['total_datasets']}",
                f"  Total Examples: {analysis['summary']['total_examples']:,}",
                f"  Total Size: {analysis['summary']['total_size_mb']:.1f} MB",
                "",
                "BY TASK TYPE:"
            ]
            
            for task_type, stats in datasets_by_task.items():
                report_lines.extend([
                    f"  {task_type.upper()}:",
                    f"    Datasets: {stats['count']}",
                    f"    Examples: {stats['examples']:,}",
                    f"    Size: {stats['size_mb']:.1f} MB"
                ])
            
            report_lines.extend([
                "",
                "DATASET DETAILS:",
                ""
            ])
            
            for dataset_name, info in analysis['datasets'].items():
                report_lines.extend([
                    f"  {dataset_name}:",
                    f"    Task: {info['task_type']}",
                    f"    Examples: {info['num_examples']:,}",
                    f"    Size: {info['file_size_mb']:.1f} MB"
                ])
            
            report_lines.extend([
                "",
                "=" * 60,
                "RESEARCH READINESS STATUS:",
                ""
            ])
            
            # Check research readiness
            has_kg = any('kg' in info['task_type'] for info in analysis['datasets'].values())
            has_qa = any('qa' in info['task_type'] for info in analysis['datasets'].values())
            has_ir = any('ir' in info['task_type'] for info in analysis['datasets'].values())
            
            report_lines.extend([
                f"  Knowledge Graph (KG): {'✓' if has_kg else '✗'}",
                f"  Question Answering (QA): {'✓' if has_qa else '✗'}",
                f"  Information Retrieval (IR): {'✓' if has_ir else '✗'}",
                "",
                f"  KG+Dense Complementarity Research: {'READY ✓' if (has_kg and has_qa and has_ir) else 'NOT READY ✗'}",
                "=" * 60
            ])
            
            report_content = '\n'.join(report_lines)
            
            # Save report
            report_file = self.analysis_dir / "dataset_analysis_report.md"
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            # Print report
            print(report_content)
            
            logger.info(f"Analysis complete. Report saved to {report_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error in validation and analysis: {e}")
            return False
    
    def run_comprehensive_fix(self) -> bool:
        """Run all dataset fixing phases"""
        logger.info("Starting comprehensive dataset fixing...")
        
        results = {
            'wikidata5m': False,
            'natural_questions': False,
            'beir': False,
            'analysis': False
        }
        
        # Phase 1: Fix Wikidata5M (Critical)
        results['wikidata5m'] = self.fix_wikidata5m()
        
        # Phase 2: Process Natural Questions
        results['natural_questions'] = self.process_natural_questions()
        
        # Phase 3: Fix BEIR datasets
        results['beir'] = self.fix_beir_datasets()
        
        # Phase 4: Validation and Analysis
        results['analysis'] = self.validate_and_analyze()
        
        # Summary
        success_count = sum(results.values())
        logger.info(f"Dataset fixing complete: {success_count}/4 phases successful")
        
        if results['wikidata5m'] and (results['natural_questions'] or results['beir']):
            logger.info("✓ CRITICAL: KG dataset fixed - KG+Dense research can proceed!")
        else:
            logger.warning("✗ CRITICAL: KG dataset not fixed - KG+Dense research blocked!")
        
        return success_count >= 3

def main():
    """Main execution function"""
    print("=" * 60)
    print("COMPREHENSIVE DATASET FIXING FOR KG + DENSE VECTOR RESEARCH")
    print("=" * 60)
    
    fixer = ComprehensiveDatasetFixer()
    success = fixer.run_comprehensive_fix()
    
    if success:
        print("\n🎉 Dataset fixing completed successfully!")
        print("Ready to proceed with KG + Dense Vector complementarity research!")
    else:
        print("\n❌ Dataset fixing encountered issues.")
        print("Check logs for details and retry failed components.")
    
    return success

if __name__ == "__main__":
    main()
