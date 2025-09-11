#!/usr/bin/env python3
import json
import argparse
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser(description="Validate a JSONL file.")
    ap.add_argument("in_jsonl", help="Path to the JSONL file to validate")
    args = ap.parse_args()

    bad_lines = 0
    total_lines = 0

    print(f"Validating {args.in_jsonl}...")
    with open(args.in_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Checking lines")):
            total_lines += 1
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error on line {i+1}: {e}")
                bad_lines += 1
    
    print("\nValidation complete.")
    print(f"Total lines: {total_lines}")
    if bad_lines == 0:
        print("✅ All lines are valid JSON.")
    else:
        print(f"❌ Found {bad_lines} invalid JSON lines.")

if __name__ == "__main__":
    main()
