#!/usr/bin/env python3
"""
Process full datasets for SOTA-level KG + Dense Vector research
Efficient processing for M1 MacBook with full dataset integrity
"""

import os
import json
import logging
from pathlib import Path
from datasets import load_from_disk, Dataset
import pandas as pd
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FullDatasetProcessor:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create processed directories
        for subdir in ["qa", "ir", "kg"]:
            (self.processed_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def process_natural_questions(self):
        """Process full Natural Questions dataset"""
        logger.info("Processing Natural Questions dataset...")
        
        nq_raw_dir = self.raw_dir / "qa" / "natural_questions"
        nq_processed_dir = self.processed_dir / "qa" / "natural_questions"
        
        if not nq_raw_dir.exists():
            logger.error("Natural Questions raw data not found")
            return False
        
        try:
            # Load the dataset
            dataset = load_from_disk(str(nq_raw_dir))
            logger.info(f"Loaded NQ dataset: {dataset}")
            
            # Process each split
            processed_data = {}
            for split_name, split_data in dataset.items():
                logger.info(f"Processing {split_name} split: {len(split_data)} examples")
                
                processed_examples = []
                for example in tqdm(split_data, desc=f"Processing {split_name}"):
                    # Extract question-answer pairs
                    if 'question' in example and 'annotations' in example:
                        question = example['question']['text']
                        
                        # Extract short answers
                        short_answers = []
                        if example['annotations']:
                            for annotation in example['annotations']:
                                if annotation.get('short_answers'):
                                    for short_answer in annotation['short_answers']:
                                        if 'text' in short_answer:
                                            short_answers.append(short_answer['text'])
                        
                        if short_answers:  # Only keep examples with answers
                            processed_examples.append({
                                'question': question,
                                'answers': short_answers,
                                'example_id': example.get('example_id', ''),
                                'document_title': example.get('document_title', ''),
                                'document_url': example.get('document_url', '')
                            })
                
                processed_data[split_name] = processed_examples
                logger.info(f"Processed {split_name}: {len(processed_examples)} QA pairs")
            
            # Save processed data
            for split_name, data in processed_data.items():
                output_file = nq_processed_dir / f"{split_name}.jsonl"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w') as f:
                    for item in data:
                        f.write(json.dumps(item) + '\n')
                
                logger.info(f"Saved {len(data)} {split_name} examples to {output_file}")
            
            # Create summary
            summary = {
                'dataset': 'natural_questions',
                'splits': {name: len(data) for name, data in processed_data.items()},
                'total_examples': sum(len(data) for data in processed_data.values()),
                'processing_date': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            summary_file = nq_processed_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"✓ Natural Questions processing completed: {summary['total_examples']} total examples")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to process Natural Questions: {e}")
            return False
    
    def process_ms_marco_qa(self):
        """Process full MS MARCO QA dataset"""
        logger.info("Processing MS MARCO QA dataset...")
        
        marco_raw_dir = self.raw_dir / "qa" / "ms_marco_qa"
        marco_processed_dir = self.processed_dir / "qa" / "ms_marco_qa"
        
        if not marco_raw_dir.exists():
            logger.error("MS MARCO QA raw data not found")
            return False
        
        try:
            # Load the dataset
            dataset = load_from_disk(str(marco_raw_dir))
            logger.info(f"Loaded MS MARCO QA dataset: {dataset}")
            
            # Process each split
            processed_data = {}
            for split_name, split_data in dataset.items():
                logger.info(f"Processing {split_name} split: {len(split_data)} examples")
                
                processed_examples = []
                for example in tqdm(split_data, desc=f"Processing {split_name}"):
                    if 'query' in example and 'answers' in example:
                        # Extract answers
                        answers = example['answers']
                        if answers and any(answers):  # Only keep examples with non-empty answers
                            processed_examples.append({
                                'question': example['query'],
                                'answers': [ans for ans in answers if ans],  # Filter empty answers
                                'query_id': example.get('query_id', ''),
                                'query_type': example.get('query_type', ''),
                                'wellFormedAnswers': example.get('wellFormedAnswers', [])
                            })
                
                processed_data[split_name] = processed_examples
                logger.info(f"Processed {split_name}: {len(processed_examples)} QA pairs")
            
            # Save processed data
            for split_name, data in processed_data.items():
                output_file = marco_processed_dir / f"{split_name}.jsonl"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w') as f:
                    for item in data:
                        f.write(json.dumps(item) + '\n')
                
                logger.info(f"Saved {len(data)} {split_name} examples to {output_file}")
            
            # Create summary
            summary = {
                'dataset': 'ms_marco_qa',
                'splits': {name: len(data) for name, data in processed_data.items()},
                'total_examples': sum(len(data) for data in processed_data.values()),
                'processing_date': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            summary_file = marco_processed_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"✓ MS MARCO QA processing completed: {summary['total_examples']} total examples")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to process MS MARCO QA: {e}")
            return False
    
    def process_ms_marco_passage(self):
        """Process full MS MARCO Passage dataset"""
        logger.info("Processing MS MARCO Passage dataset...")
        
        passage_raw_dir = self.raw_dir / "ir" / "ms_marco_passage"
        passage_processed_dir = self.processed_dir / "ir" / "ms_marco_passage"
        
        if not passage_raw_dir.exists():
            logger.error("MS MARCO Passage raw data not found")
            return False
        
        try:
            # Load the dataset
            dataset = load_from_disk(str(passage_raw_dir))
            logger.info(f"Loaded MS MARCO Passage dataset: {dataset}")
            
            # Process each split
            processed_data = {}
            for split_name, split_data in dataset.items():
                logger.info(f"Processing {split_name} split: {len(split_data)} examples")
                
                processed_examples = []
                for example in tqdm(split_data, desc=f"Processing {split_name}"):
                    if 'query' in example:
                        processed_examples.append({
                            'query': example['query'],
                            'query_id': example.get('query_id', ''),
                            'passage': example.get('passage', ''),
                            'passage_id': example.get('passage_id', ''),
                            'relevance': example.get('relevance', 0)
                        })
                
                processed_data[split_name] = processed_examples
                logger.info(f"Processed {split_name}: {len(processed_examples)} query-passage pairs")
            
            # Save processed data
            for split_name, data in processed_data.items():
                output_file = passage_processed_dir / f"{split_name}.jsonl"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w') as f:
                    for item in data:
                        f.write(json.dumps(item) + '\n')
                
                logger.info(f"Saved {len(data)} {split_name} examples to {output_file}")
            
            # Create summary
            summary = {
                'dataset': 'ms_marco_passage',
                'splits': {name: len(data) for name, data in processed_data.items()},
                'total_examples': sum(len(data) for data in processed_data.values()),
                'processing_date': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            summary_file = passage_processed_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"✓ MS MARCO Passage processing completed: {summary['total_examples']} total examples")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to process MS MARCO Passage: {e}")
            return False
    
    def process_wikidata5m(self):
        """Process Wikidata5M knowledge graph"""
        logger.info("Processing Wikidata5M dataset...")
        
        kg_raw_dir = self.raw_dir / "kg" / "wikidata5m"
        kg_processed_dir = self.processed_dir / "kg" / "wikidata5m"
        
        if not kg_raw_dir.exists():
            logger.error("Wikidata5M raw data not found")
            return False
        
        try:
            import gzip
            
            # Process entities
            entities_file = kg_raw_dir / "entities.txt.gz"
            if entities_file.exists():
                entities = []
                with gzip.open(entities_file, 'rt') as f:
                    for line in tqdm(f, desc="Processing entities"):
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            entities.append({
                                'entity_id': parts[0],
                                'entity_name': parts[1]
                            })
                
                # Save entities
                entities_output = kg_processed_dir / "entities.jsonl"
                entities_output.parent.mkdir(parents=True, exist_ok=True)
                
                with open(entities_output, 'w') as f:
                    for entity in entities:
                        f.write(json.dumps(entity) + '\n')
                
                logger.info(f"Processed {len(entities)} entities")
            
            # Process relations
            relations_file = kg_raw_dir / "relations.txt.gz"
            if relations_file.exists():
                relations = []
                with gzip.open(relations_file, 'rt') as f:
                    for line in tqdm(f, desc="Processing relations"):
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            relations.append({
                                'relation_id': parts[0],
                                'relation_name': parts[1]
                            })
                
                # Save relations
                relations_output = kg_processed_dir / "relations.jsonl"
                with open(relations_output, 'w') as f:
                    for relation in relations:
                        f.write(json.dumps(relation) + '\n')
                
                logger.info(f"Processed {len(relations)} relations")
            
            # Process triples
            for split in ['train', 'valid', 'test']:
                triples_file = kg_raw_dir / f"triples_{split}.txt.gz"
                if triples_file.exists():
                    triples = []
                    with gzip.open(triples_file, 'rt') as f:
                        for line in tqdm(f, desc=f"Processing {split} triples"):
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                triples.append({
                                    'head': parts[0],
                                    'relation': parts[1],
                                    'tail': parts[2]
                                })
                    
                    # Save triples
                    triples_output = kg_processed_dir / f"triples_{split}.jsonl"
                    with open(triples_output, 'w') as f:
                        for triple in triples:
                            f.write(json.dumps(triple) + '\n')
                    
                    logger.info(f"Processed {len(triples)} {split} triples")
            
            logger.info("✓ Wikidata5M processing completed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to process Wikidata5M: {e}")
            return False
    
    def create_processing_summary(self):
        """Create overall processing summary"""
        summary = {
            'processing_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'datasets': {}
        }
        
        # Check each processed dataset
        for task in ['qa', 'ir', 'kg']:
            task_dir = self.processed_dir / task
            if task_dir.exists():
                for dataset_dir in task_dir.iterdir():
                    if dataset_dir.is_dir():
                        summary_file = dataset_dir / "summary.json"
                        if summary_file.exists():
                            with open(summary_file, 'r') as f:
                                dataset_summary = json.load(f)
                                summary['datasets'][dataset_dir.name] = dataset_summary
        
        # Save overall summary
        summary_file = self.processed_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    """Main processing function"""
    processor = FullDatasetProcessor()
    
    logger.info("Starting full dataset processing for SOTA research...")
    
    # Process all datasets
    results = {}
    results['natural_questions'] = processor.process_natural_questions()
    results['ms_marco_qa'] = processor.process_ms_marco_qa()
    results['ms_marco_passage'] = processor.process_ms_marco_passage()
    results['wikidata5m'] = processor.process_wikidata5m()
    
    # Create summary
    summary = processor.create_processing_summary()
    
    print("\n" + "="*60)
    print("FULL DATASET PROCESSING COMPLETED!")
    print("="*60)
    
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"Successfully processed: {successful}/{total} datasets")
    
    for dataset, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {dataset}")
    
    if 'datasets' in summary:
        print(f"\nTotal examples processed:")
        for dataset_name, dataset_info in summary['datasets'].items():
            if 'total_examples' in dataset_info:
                print(f"  • {dataset_name}: {dataset_info['total_examples']:,} examples")
    
    print(f"\nProcessed data location: {processor.processed_dir}")
    print("\nReady for SOTA-level model development and evaluation!")

if __name__ == "__main__":
    main()
