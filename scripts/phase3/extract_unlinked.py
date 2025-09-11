#!/usr/bin/env python3
import json
import argparse
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser(description="Extract samples with no entity linking results from a JSONL file.")
    ap.add_argument("--in_jsonl", required=True, help="Input JSONL file")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL file for unlinked samples")
    args = ap.parse_args()

    count = 0
    with open(args.in_jsonl, 'r', encoding='utf-8') as f_in, open(args.out_jsonl, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc="Finding unlinked samples"):
            sample = json.loads(line)
            if not sample.get("el_results"):
                f_out.write(line)
                count += 1

    print(f"Extracted {count} unlinked samples to {args.out_jsonl}")

if __name__ == "__main__":
    main()
