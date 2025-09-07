#!/usr/bin/env python3
"""
MS MARCO QA Dataset Download and Preprocessing
Downloads MS MARCO QA dataset and prepares it for QA experiments
"""

import os
import json
import logging
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    base_dir = Path.cwd()
    raw_dir = base_dir / "data" / "raw" / "qa" / "ms_marco_qa"
    processed_dir = base_dir / "data" / "processed" / "qa" / "ms_marco_qa"
    log_dir = base_dir / "logs" / "preprocessing"
    
    for dir_path in [raw_dir, processed_dir, log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return raw_dir, processed_dir, log_dir

def download_ms_marco_qa(raw_dir):
    """Download MS MARCO QA dataset using HuggingFace datasets"""
    logger.info("Starting MS MARCO QA download...")
    
    try:
        # Load dataset from HuggingFace
        logger.info("Loading MS MARCO QA from HuggingFace...")
        dataset = load_dataset("ms_marco", "v2.1", cache_dir=raw_dir / "cache")
        
        # Save raw data
        logger.info("Saving raw dataset files...")
        for split in dataset.keys():
            output_file = raw_dir / f"ms_marco_qa-{split}.jsonl"
            with open(output_file, 'w') as f:
                for example in tqdm(dataset[split], desc=f"Saving {split}"):
                    f.write(json.dumps(example) + '\n')
            logger.info(f"Saved {split} split: {len(dataset[split])} examples")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error downloading MS MARCO QA: {e}")
        return None

def extract_qa_pairs(example):
    """Extract question-answer pairs from MS MARCO example"""
    
    # MS MARCO has different structure than Natural Questions
    question = example.get('query', '')
    answers = example.get('answers', [])
    passages = example.get('passages', [])
    
    # Extract answer text
    answer_texts = []
    if answers:
        for answer in answers:
            if isinstance(answer, str):
                answer_texts.append(answer)
            elif isinstance(answer, dict) and 'text' in answer:
                answer_texts.append(answer['text'])
    
    # Extract relevant passages
    relevant_passages = []
    if passages:
        for passage in passages:
            if isinstance(passage, dict):
                passage_text = passage.get('passage_text', '')
                is_selected = passage.get('is_selected', 0)
                if passage_text and is_selected:
                    relevant_passages.append(passage_text)
    
    return {
        'question_id': example.get('query_id', ''),
        'question': question,
        'answers': answer_texts,
        'passages': relevant_passages,
        'has_answer': len(answer_texts) > 0,
        'query_type': example.get('query_type', 'unknown')
    }

def preprocess_dataset(dataset, processed_dir):
    """Preprocess MS MARCO QA for QA task"""
    logger.info("Starting preprocessing...")
    
    processed_data = {}
    
    for split in dataset.keys():
        logger.info(f"Processing {split} split...")
        split_data = []
        
        for example in tqdm(dataset[split], desc=f"Processing {split}"):
            qa_pair = extract_qa_pairs(example)
            
            # Only keep examples with answers for training
            if split == 'train' and not qa_pair['has_answer']:
                continue
                
            split_data.append(qa_pair)
        
        processed_data[split] = split_data
        logger.info(f"Processed {len(split_data)} examples for {split}")
        
        # Save processed split
        output_file = processed_dir / f"ms_marco_qa-{split}-processed.jsonl"
        with open(output_file, 'w') as f:
            for item in split_data:
                f.write(json.dumps(item) + '\n')
    
    # Create summary statistics
    stats = {
        'dataset': 'MS MARCO QA',
        'splits': {},
        'preprocessing_config': {
            'keep_unanswered': False,
            'extract_answers': True,
            'extract_passages': True
        }
    }
    
    for split, data in processed_data.items():
        if data:
            stats['splits'][split] = {
                'total_examples': len(data),
                'with_answers': sum(1 for x in data if x['answers']),
                'with_passages': sum(1 for x in data if x['passages']),
                'avg_question_length': sum(len(x['question'].split()) for x in data) / len(data),
                'avg_answers_per_question': sum(len(x['answers']) for x in data) / len(data)
            }
    
    # Save statistics
    stats_file = processed_dir / "preprocessing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Preprocessing complete!")
    return processed_data, stats

def create_sample_subset(processed_dir, sample_size=1000):
    """Create small sample for quick testing"""
    logger.info(f"Creating sample subset of {sample_size} examples...")
    
    # Load processed data
    train_file = processed_dir / "ms_marco_qa-train-processed.jsonl"
    if not train_file.exists():
        logger.error("Processed training data not found")
        return
    
    # Read and sample
    train_data = []
    with open(train_file, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    # Take sample
    sample_data = train_data[:sample_size]
    
    # Save sample
    sample_file = processed_dir / f"ms_marco_qa-train-sample-{sample_size}.jsonl"
    with open(sample_file, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Created sample dataset: {sample_file}")

def main():
    """Main processing function"""
    logger.info("Starting MS MARCO QA download and preprocessing")
    
    # Setup directories
    raw_dir, processed_dir, log_dir = setup_directories()
    
    # Setup file logging
    log_file = log_dir / "ms_marco_qa_processing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    try:
        # Download dataset
        dataset = download_ms_marco_qa(raw_dir)
        if dataset is None:
            logger.error("Failed to download dataset")
            return False
        
        # Preprocess dataset
        processed_data, stats = preprocess_dataset(dataset, processed_dir)
        
        # Create sample subset
        create_sample_subset(processed_dir)
        
        # Print summary
        print("\n" + "="*50)
        print("MS MARCO QA Processing Complete!")
        print("="*50)
        print(f"Raw data saved to: {raw_dir}")
        print(f"Processed data saved to: {processed_dir}")
        print(f"Processing log: {log_file}")
        print("\nDataset Statistics:")
        for split, split_stats in stats['splits'].items():
            print(f"  {split.capitalize()}:")
            print(f"    Total examples: {split_stats['total_examples']:,}")
            print(f"    With answers: {split_stats['with_answers']:,}")
            print(f"    With passages: {split_stats['with_passages']:,}")
            print(f"    Avg question length: {split_stats['avg_question_length']:.1f} words")
            print(f"    Avg answers per question: {split_stats['avg_answers_per_question']:.1f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
