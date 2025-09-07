#!/usr/bin/env python3
"""
M1-Optimized Sample Dataset Creator
Creates small representative samples for local development
"""

import json
import logging
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_nq_sample(sample_size=5000):
    """Create Natural Questions sample dataset"""
    logger.info(f"Creating Natural Questions sample ({sample_size} examples)...")
    
    # Load only validation split (much smaller)
    dataset = load_dataset("natural_questions", split="validation[:10000]")
    
    processed_data = []
    for i, example in enumerate(tqdm(dataset, desc="Processing NQ")):
        if len(processed_data) >= sample_size:
            break
            
        question = example['question']['text']
        
        # Extract answers
        short_answers = []
        long_answers = []
        
        if example['annotations']:
            for annotation in example['annotations']:
                if annotation['short_answers']:
                    for short_answer in annotation['short_answers']:
                        start_token = short_answer['start_token']
                        end_token = short_answer['end_token']
                        if 'document' in example and 'tokens' in example['document']:
                            tokens = example['document']['tokens']
                            if start_token < len(tokens) and end_token <= len(tokens):
                                answer_tokens = tokens[start_token:end_token]
                                answer_text = ' '.join([token['token'] for token in answer_tokens])
                                short_answers.append(answer_text)
                
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
        
        # Only keep examples with answers
        if short_answers or long_answers:
            processed_data.append({
                'question_id': example['id'],
                'question': question,
                'short_answers': short_answers,
                'long_answers': long_answers,
                'document_title': example['document']['title'] if 'document' in example else '',
                'has_answer': True
            })
    
    return processed_data

def create_msmarco_sample(sample_size=5000):
    """Create MS MARCO sample dataset"""
    logger.info(f"Creating MS MARCO sample ({sample_size} examples)...")
    
    try:
        # Load small sample from dev split
        dataset = load_dataset("ms_marco", "v2.1", split="validation[:10000]")
        
        processed_data = []
        for example in tqdm(dataset, desc="Processing MS MARCO"):
            if len(processed_data) >= sample_size:
                break
                
            question = example.get('query', '')
            answers = example.get('answers', [])
            
            if question and answers:
                processed_data.append({
                    'question_id': example.get('query_id', ''),
                    'question': question,
                    'answers': answers,
                    'has_answer': True
                })
        
        return processed_data
    except Exception as e:
        logger.warning(f"MS MARCO sample creation failed: {e}")
        return []

def create_beir_samples():
    """Create small BEIR samples for IR tasks"""
    logger.info("Creating BEIR samples...")
    
    beir_datasets = ['nfcorpus', 'scifact']  # Start with smallest ones
    samples = {}
    
    for dataset_name in beir_datasets:
        try:
            # Load small samples
            dataset = load_dataset("BeIR/beir", dataset_name, split="test[:500]")
            
            processed_data = []
            for example in tqdm(dataset, desc=f"Processing {dataset_name}"):
                processed_data.append({
                    'query_id': example.get('_id', ''),
                    'query': example.get('text', ''),
                    'title': example.get('title', ''),
                    'metadata': example.get('metadata', {})
                })
            
            samples[dataset_name] = processed_data
            logger.info(f"Created {len(processed_data)} samples for {dataset_name}")
            
        except Exception as e:
            logger.warning(f"Failed to create {dataset_name} sample: {e}")
    
    return samples

def save_samples():
    """Save all sample datasets"""
    base_dir = Path.cwd()
    
    # Create sample directories
    sample_dir = base_dir / "data" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Natural Questions sample
    nq_data = create_nq_sample(5000)
    nq_file = sample_dir / "natural_questions_sample.jsonl"
    with open(nq_file, 'w') as f:
        for item in nq_data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved {len(nq_data)} NQ samples to {nq_file}")
    
    # Create MS MARCO sample
    marco_data = create_msmarco_sample(5000)
    if marco_data:
        marco_file = sample_dir / "ms_marco_qa_sample.jsonl"
        with open(marco_file, 'w') as f:
            for item in marco_data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved {len(marco_data)} MS MARCO samples to {marco_file}")
    
    # Create BEIR samples
    beir_data = create_beir_samples()
    for dataset_name, data in beir_data.items():
        beir_file = sample_dir / f"beir_{dataset_name}_sample.jsonl"
        with open(beir_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved {len(data)} {dataset_name} samples to {beir_file}")
    
    # Create summary
    summary = {
        "natural_questions": len(nq_data),
        "ms_marco_qa": len(marco_data),
        "beir_datasets": {name: len(data) for name, data in beir_data.items()},
        "total_size_mb": sum(f.stat().st_size for f in sample_dir.glob("*.jsonl")) / (1024*1024),
        "description": "Lightweight samples for M1 development and testing"
    }
    
    summary_file = sample_dir / "sample_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*50)
    print("Sample Datasets Created!")
    print("="*50)
    print(f"Location: {sample_dir}")
    print(f"Natural Questions: {len(nq_data):,} examples")
    print(f"MS MARCO QA: {len(marco_data):,} examples")
    for name, data in beir_data.items():
        print(f"BEIR {name}: {len(data):,} examples")
    print(f"Total size: {summary['total_size_mb']:.1f} MB")
    print("\nThese samples are perfect for:")
    print("- Model development and testing")
    print("- Baseline implementation")
    print("- Fusion architecture prototyping")
    print("- Quick experiments on M1")

if __name__ == "__main__":
    save_samples()
