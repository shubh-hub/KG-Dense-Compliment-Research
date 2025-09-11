#!/usr/bin/env python3
import json
import argparse
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser(description="Merge repaired samples back into the main dataset.")
    ap.add_argument("--original_jsonl", required=True, help="Original full dataset")
    ap.add_argument("--repaired_jsonl", required=True, help="Repaired unlinked samples")
    ap.add_argument("--out_jsonl", required=True, help="Output file for the merged dataset")
    args = ap.parse_args()

    repaired_samples = {}
    with open(args.repaired_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            repaired_samples[sample['id']] = sample

    print(f"Loaded {len(repaired_samples)} repaired samples.")

    with open(args.original_jsonl, 'r', encoding='utf-8') as f_in, open(args.out_jsonl, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc="Merging samples"):
            original_sample = json.loads(line)
            sample_id = original_sample['id']
            
            if sample_id in repaired_samples:
                f_out.write(json.dumps(repaired_samples[sample_id], ensure_ascii=False) + '\n')
            else:
                f_out.write(line)

    print(f"Merged dataset saved to {args.out_jsonl}")

if __name__ == "__main__":
    main()
