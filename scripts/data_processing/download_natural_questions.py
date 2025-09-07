#!/usr/bin/env python3
"""
Natural Questions Dataset Download and Preprocessing
Downloads NQ dataset and prepares it for QA experiments
"""

import os
import json
import gzip
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
    raw_dir = base_dir / "data" / "raw" / "qa" / "natural_questions"
    processed_dir = base_dir / "data" / "processed" / "qa" / "natural_questions"
    log_dir = base_dir / "logs" / "preprocessing"
    
    for dir_path in [raw_dir, processed_dir, log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return raw_dir, processed_dir, log_dir

def download_natural_questions(raw_dir):
    """Download Natural Questions dataset using HuggingFace datasets"""
    logger.info("Starting Natural Questions download...")
    
    try:
        # Load dataset from HuggingFace
        logger.info("Loading Natural Questions from HuggingFace...")
        dataset = load_dataset("natural_questions", cache_dir=raw_dir / "cache")
        
        # Save raw data
        logger.info("Saving raw dataset files...")
        for split in dataset.keys():
            output_file = raw_dir / f"nq-{split}.jsonl"
            with open(output_file, 'w') as f:
                for example in tqdm(dataset[split], desc=f"Saving {split}"):
                    f.write(json.dumps(example) + '\n')
            logger.info(f"Saved {split} split: {len(dataset[split])} examples")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error downloading Natural Questions: {e}")
        return None

def extract_qa_pairs(example):
    """Extract question-answer pairs from NQ example"""
    question = example['question']['text']
    
    # Extract short answers
    short_answers = []
    if example['annotations']:
        for annotation in example['annotations']:
            if annotation['short_answers']:
                for short_answer in annotation['short_answers']:
                    start_token = short_answer['start_token']
                    end_token = short_answer['end_token']
                    # Extract text from document tokens
                    if 'document' in example and 'tokens' in example['document']:
                        tokens = example['document']['tokens']
                        if start_token < len(tokens) and end_token <= len(tokens):
                            answer_tokens = tokens[start_token:end_token]
                            answer_text = ' '.join([token['token'] for token in answer_tokens])
                            short_answers.append(answer_text)
    
    # Extract long answers (passages)
    long_answers = []
    if example['annotations']:
        for annotation in example['annotations']:
            if annotation['long_answer']:
                long_answer = annotation['long_answer']
                start_token = long_answer['start_token']
                end_token = long_answer['end_token']
                if 'document' in example and 'tokens' in example['document']:
                    tokens = example['document']['tokens']
                    if start_token < len(tokens) and end_token <= len(tokens):
                        passage_tokens = tokens[start_token:end_token]
                        passage_text = ' '.join([token['token'] for token in passage_tokens])
                        long_answers.append(passage_text)
    
    return {
        'question_id': example['id'],
        'question': question,
        'short_answers': short_answers,
        'long_answers': long_answers,
        'document_title': example['document']['title'] if 'document' in example else '',
        'has_answer': len(short_answers) > 0 or len(long_answers) > 0
    }

def preprocess_dataset(dataset, processed_dir):
    """Preprocess Natural Questions for QA task"""
    logger.info("Starting preprocessing...")
    
    processed_data = {'train': [], 'validation': []}
    
    for split in ['train', 'validation']:
        if split not in dataset:
            logger.warning(f"Split {split} not found in dataset")
            continue
            
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
        output_file = processed_dir / f"nq-{split}-processed.jsonl"
        with open(output_file, 'w') as f:
            for item in split_data:
                f.write(json.dumps(item) + '\n')
    
    # Create summary statistics
    stats = {
        'dataset': 'Natural Questions',
        'splits': {},
        'preprocessing_config': {
            'keep_unanswered': False,
            'extract_short_answers': True,
            'extract_long_answers': True
        }
    }
    
    for split, data in processed_data.items():
        stats['splits'][split] = {
            'total_examples': len(data),
            'with_short_answers': sum(1 for x in data if x['short_answers']),
            'with_long_answers': sum(1 for x in data if x['long_answers']),
            'avg_question_length': sum(len(x['question'].split()) for x in data) / len(data) if data else 0
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
    train_file = processed_dir / "nq-train-processed.jsonl"
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
    sample_file = processed_dir / f"nq-train-sample-{sample_size}.jsonl"
    with open(sample_file, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Created sample dataset: {sample_file}")

def main():
    """Main processing function"""
    logger.info("Starting Natural Questions download and preprocessing")
    
    # Setup directories
    raw_dir, processed_dir, log_dir = setup_directories()
    
    # Setup file logging
    log_file = log_dir / "natural_questions_processing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    try:
        # Download dataset
        dataset = download_natural_questions(raw_dir)
        if dataset is None:
            logger.error("Failed to download dataset")
            return False
        
        # Preprocess dataset
        processed_data, stats = preprocess_dataset(dataset, processed_dir)
        
        # Create sample subset
        create_sample_subset(processed_dir)
        
        # Print summary
        print("\n" + "="*50)
        print("Natural Questions Processing Complete!")
        print("="*50)
        print(f"Raw data saved to: {raw_dir}")
        print(f"Processed data saved to: {processed_dir}")
        print(f"Processing log: {log_file}")
        print("\nDataset Statistics:")
        for split, split_stats in stats['splits'].items():
            print(f"  {split.capitalize()}:")
            print(f"    Total examples: {split_stats['total_examples']:,}")
            print(f"    With short answers: {split_stats['with_short_answers']:,}")
            print(f"    With long answers: {split_stats['with_long_answers']:,}")
            print(f"    Avg question length: {split_stats['avg_question_length']:.1f} words")
        
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
