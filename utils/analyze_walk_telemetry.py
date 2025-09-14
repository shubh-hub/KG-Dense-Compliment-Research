print("[DEBUG] Script started.")
import json
import numpy as np
import argparse
from collections import Counter

def analyze_walk(file_path):
    hits_by_hop = Counter()
    seeds_vs_success = {'hit': [], 'miss': []}
    allowlist_usage = {'enabled': 0, 'disabled': 0, 'enabled_hit': 0, 'disabled_hit': 0}
    total_samples = 0

    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip(): continue
            s = json.loads(line)
            if s.get('source') != 'kg-qa-generated': continue
            total_samples += 1

            gold = s.get('motif', [None]*5)[-1]
            kg_cands = s.get('kg_candidates', [])
            
            # 1. Seeds vs. Success
            num_seeds = len(s.get('seeds', []))
            is_hit = any(c['qid'] == gold for c in kg_cands[:20])
            seeds_vs_success['hit' if is_hit else 'miss'].append(num_seeds)

            # 2. Allowlist Usage
            allowlist_enabled = s.get('allowlist_meta', {}).get('enabled', False)
            if allowlist_enabled:
                allowlist_usage['enabled'] += 1
                if is_hit: allowlist_usage['enabled_hit'] += 1
            else:
                allowlist_usage['disabled'] += 1
                if is_hit: allowlist_usage['disabled_hit'] += 1

            # 3. Hit by Hop
            if is_hit:
                for cand in kg_cands:
                    if cand['qid'] == gold:
                        hits_by_hop[cand['hop']] += 1
                        break

    print("--- Walk Telemetry Analysis ---")
    print(f"Analyzed {total_samples} samples from {file_path}\n")

    # Print Seeds vs. Success
    avg_seeds_hit = np.mean(seeds_vs_success['hit']) if seeds_vs_success['hit'] else 0
    avg_seeds_miss = np.mean(seeds_vs_success['miss']) if seeds_vs_success['miss'] else 0
    print(f"[Seeds vs. Success]")
    print(f"  - Avg seeds for HIT: {avg_seeds_hit:.2f}")
    print(f"  - Avg seeds for MISS: {avg_seeds_miss:.2f}\n")

    # Print Allowlist Usage
    enabled_rate = allowlist_usage['enabled'] / total_samples if total_samples > 0 else 0
    enabled_hit_rate = allowlist_usage['enabled_hit'] / allowlist_usage['enabled'] if allowlist_usage['enabled'] > 0 else 0
    disabled_hit_rate = allowlist_usage['disabled_hit'] / allowlist_usage['disabled'] if allowlist_usage['disabled'] > 0 else 0
    print("[Allowlist Usage]")
    print(f"  - % Queries with Allowlist Enabled: {enabled_rate:.2%}")
    print(f"  - Hit Rate with Allowlist: {enabled_hit_rate:.2%}")
    print(f"  - Hit Rate without Allowlist (safety valve): {disabled_hit_rate:.2%}\n")

    # Print Hits by Hop
    total_hits = sum(hits_by_hop.values())
    print("[Hits by Hop]")
    if total_hits > 0:
        for hop in sorted(hits_by_hop.keys()):
            perc = hits_by_hop[hop] / total_hits
            print(f"  - Hop {hop}: {hits_by_hop[hop]} hits ({perc:.2%})")
    else:
        print("  - No hits found.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze KG walker telemetry.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the walked JSONL file.')
    args = parser.parse_args()
    analyze_walk(args.file_path)
