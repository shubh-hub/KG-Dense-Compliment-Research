#!/usr/bin/env python3
"""
Combine all generated sample files into one large corpus.
"""

import json
from pathlib import Path

def main():
    sample_dir = Path("data/phase2/samples")
    output_file = sample_dir / "combined_12k.jsonl"
    
    files_to_combine = [
        sample_dir / "nq_sample.jsonl",
        sample_dir / "msmarco_sample.jsonl",
        sample_dir / "beir_sample.jsonl",
        sample_dir / "kgqa_sample.jsonl"
    ]
    
    all_samples = []
    for file_path in files_to_combine:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    all_samples.append(json.loads(line.strip()))
            print(f"Loaded {file_path.name}")
        else:
            print(f"Warning: {file_path.name} not found.")

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
            
    print(f"\nCombined {len(all_samples)} samples into {output_file}")

if __name__ == "__main__":
    main()
