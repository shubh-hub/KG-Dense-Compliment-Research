import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default='data/phase2/kg_subset_triples_expanded_v2.jsonl')
    parser.add_argument('--out_file', type=str, default='data/phase2/kge/train.tsv')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    print(f"Reading from {args.in_file} and writing to {args.out_file}...")
    with open(args.in_file, 'r', encoding='utf-8') as f_in, open(args.out_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                f_out.write(f"{data['head']}\t{data['relation']}\t{data['tail']}\n")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping malformed line: {e}")

    print("Done.")

if __name__ == '__main__':
    main()
