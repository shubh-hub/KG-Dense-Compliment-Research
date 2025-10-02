print("[DEBUG] Script started.")
import json
import numpy as np
import argparse
from collections import Counter

def analyze_walk(file_path):
    hits_by_hop = Counter()
    seeds_vs_success = {'hit': [], 'miss': []}
    allowlist_usage = {'enabled': 0, 'disabled': 0, 'enabled_hit': 0, 'disabled_hit': 0, 'q_only_event': 0, 's_only_event': 0, 'both_event': 0}
    allowlist_sizes = {'pre_cap': [], 'post_cap': []}
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
            meta = s.get('allowlist_meta', {})
            allowlist_enabled = meta.get('enabled', False)
            if allowlist_enabled:
                allowlist_usage['enabled'] += 1
                if is_hit: allowlist_usage['enabled_hit'] += 1
                
                allowlist_sizes['pre_cap'].append(meta.get('pre_cap_size', 0))
                allowlist_sizes['post_cap'].append(meta.get('size', 0))

                # Use boolean logic on allowlist presence
                q_present = meta.get('q_allow_size', 0) > 0
                s_present = meta.get('s_allow_size', 0) > 0
                if q_present and s_present:
                    allowlist_usage['both_event'] += 1
                elif q_present:
                    allowlist_usage['q_only_event'] += 1
                elif s_present:
                    allowlist_usage['s_only_event'] += 1
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
    if total_samples == 0:
        print("[WARN] No 'kg-qa-generated' samples found. No further analysis to report.")
        return

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
    print(f"  - Hit Rate without Allowlist: {disabled_hit_rate:.2%}")
    avg_pre_cap = np.mean(allowlist_sizes['pre_cap']) if allowlist_sizes['pre_cap'] else 0
    avg_post_cap = np.mean(allowlist_sizes['post_cap']) if allowlist_sizes['post_cap'] else 0
    print(f"  - Avg Allowlist Size: {avg_post_cap:.1f} (pre-cap: {avg_pre_cap:.1f})")

    q_event_rate = allowlist_usage['q_only_event'] / allowlist_usage['enabled'] if allowlist_usage['enabled'] > 0 else 0
    s_event_rate = allowlist_usage['s_only_event'] / allowlist_usage['enabled'] if allowlist_usage['enabled'] > 0 else 0
    both_event_rate = allowlist_usage['both_event'] / allowlist_usage['enabled'] if allowlist_usage['enabled'] > 0 else 0
    print(f"  - Composition: Q-only={q_event_rate:.2%}, S-only={s_event_rate:.2%}, Both={both_event_rate:.2%}\n")

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
