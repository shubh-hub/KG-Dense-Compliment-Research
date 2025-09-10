#!/usr/bin/env python3
"""
Sample smaller datasets for Phase 2 first pass:
- 4k Natural Questions
- 4k MS MARCO QA  
- 3k BEIR
Total: 11k + 1k KG-QA = 12k
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
import glob

def load_jsonl(file_path: str, max_items: int = None) -> List[Dict[Any, Any]]:
    """Load JSONL file with optional limit."""
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_items and i >= max_items:
                break
            try:
                items.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return items

def save_jsonl(items: List[Dict], output_path: str):
    """Save items to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def sample_natural_questions(n_samples: int = 4000) -> List[Dict]:
    """Sample from Natural Questions batches."""
    print(f"Sampling {n_samples} from Natural Questions...")
    
    # Find all NQ batch files
    nq_pattern = "data/processed/qa/natural_questions_batch_*.json"
    batch_files = glob.glob(nq_pattern)
    
    if not batch_files:
        # Try the main file
        main_file = "data/processed/qa/natural_questions.json"
        if Path(main_file).exists():
            batch_files = [main_file]
    
    all_items = []
    decoder = json.JSONDecoder()
    for batch_file in batch_files:
        with open(batch_file, 'r', encoding='utf-8') as f:
            content = f.read()
            idx = 0
            while idx < len(content):
                if len(all_items) >= n_samples * 2: # Get extra for good sampling
                    break
                try:
                    obj, end_idx = decoder.raw_decode(content[idx:].lstrip())
                    all_items.append(obj)
                    idx += end_idx
                except json.JSONDecodeError:
                    # Move to the next character and try again
                    idx += 1
        if len(all_items) >= n_samples * 2:
            break
    
    # Sample with good mix of entity-heavy queries
    random.shuffle(all_items)
    sampled = all_items[:n_samples]
    
    # Add source and normalize format
    for item in sampled:
        item['source'] = 'natural_questions'
        item['text'] = item.get('question', item.get('text', ''))
        if 'id' not in item:
            item['id'] = f"nq_{hash(item['text']) % 1000000}"
    
    print(f"Sampled {len(sampled)} Natural Questions examples")
    return sampled

def sample_ms_marco_qa(n_samples: int = 4000) -> List[Dict]:
    """Sample from MS MARCO QA."""
    print(f"Sampling {n_samples} from MS MARCO QA...")
    
    # Try different MS MARCO files
    ms_files = [
        "data/processed/qa/ms_marco_qa/ms_marco_qa-train-processed.jsonl",
        "data/processed/qa/ms_marco_qa/ms_marco_qa-validation-processed.jsonl",
        "data/processed/qa/ms_marco_qa/ms_marco_qa-dev-processed.jsonl"
    ]
    
    all_items = []
    for ms_file in ms_files:
        if Path(ms_file).exists():
            items = load_jsonl(ms_file)
            all_items.extend(items)
            if len(all_items) >= n_samples * 2:
                break
    
    # Sample short, paraphrastic queries
    random.shuffle(all_items)
    sampled = all_items[:n_samples]
    
    # Add source and normalize format
    for item in sampled:
        item['source'] = 'ms_marco_qa'
        item['text'] = item.get('query', item.get('question', item.get('text', '')))
        if 'id' not in item:
            item['id'] = f"msmarco_{hash(item['text']) % 1000000}"
    
    print(f"Sampled {len(sampled)} MS MARCO QA examples")
    return sampled

def sample_beir(n_samples: int = 3000) -> List[Dict]:
    """Sample from BEIR datasets."""
    print(f"Sampling {n_samples} from BEIR...")
    
    beir_files = [
        "data/processed/ir/beir_arguana.jsonl",
        "data/processed/ir/beir_fiqa.jsonl",
        "data/processed/ir/beir_nfcorpus.jsonl",
        "data/processed/ir/beir_scifact.jsonl"
    ]
    
    all_items = []
    for beir_file in beir_files:
        if Path(beir_file).exists():
            items = load_jsonl(beir_file)
            dataset_name = Path(beir_file).stem.replace('beir_', '')
            for item in items:
                if isinstance(item, dict):
                    item['beir_dataset'] = dataset_name
            all_items.extend(items)
    
    # Sample ranking-oriented queries
    random.shuffle(all_items)
    sampled = all_items[:n_samples]
    
    # Add source and normalize format
    for item in sampled:
        if isinstance(item, dict):
            item['source'] = 'beir'
        item['text'] = item.get('query', item.get('text', ''))
        if 'id' not in item:
            item['id'] = f"beir_{hash(item['text']) % 1000000}"
    
    print(f"Sampled {len(sampled)} BEIR examples")
    return sampled

def main():
    parser = argparse.ArgumentParser(description="Sample datasets for Phase 2")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--nq_samples", type=int, default=4000, help="Natural Questions samples")
    parser.add_argument("--msmarco_samples", type=int, default=4000, help="MS MARCO samples")
    parser.add_argument("--beir_samples", type=int, default=3000, help="BEIR samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    
    # Sample each dataset
    nq_samples = sample_natural_questions(args.nq_samples)
    msmarco_samples = sample_ms_marco_qa(args.msmarco_samples)
    beir_samples = sample_beir(args.beir_samples)
    
    # Save individual samples
    save_jsonl(nq_samples, f"{args.out_dir}/nq_sample.jsonl")
    save_jsonl(msmarco_samples, f"{args.out_dir}/msmarco_sample.jsonl")
    save_jsonl(beir_samples, f"{args.out_dir}/beir_sample.jsonl")
    
    # Combine all samples
    all_samples = nq_samples + msmarco_samples + beir_samples
    save_jsonl(all_samples, f"{args.out_dir}/combined_samples.jsonl")
    
    print(f"\nSampling complete:")
    print(f"- Natural Questions: {len(nq_samples)}")
    print(f"- MS MARCO QA: {len(msmarco_samples)}")
    print(f"- BEIR: {len(beir_samples)}")
    print(f"- Total: {len(all_samples)}")
    print(f"- Saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
