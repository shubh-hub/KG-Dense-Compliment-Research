import json
import numpy as np
import argparse
from math import comb
from collections import Counter

# --- Constants ---
DATA_PATH = "data/phase3/annotated_12k_walked_b200_h4.jsonl"
FUSED_PATH = "data/phase3/annotated_12k_fused_b200_h4.jsonl"
K = 20
N_BOOTSTRAP = 1000
N_RANDOMIZATION = 10000

# --- Data Loading ---
def load_data(data_path, fused_path):
    """Loads and prepares all necessary data from files."""
    samples = []
    fused_map = {json.loads(ln)['id']: [c['qid'] for c in json.loads(ln).get('fused_candidates', [])] for ln in open(fused_path, encoding='utf-8-sig') if ln.strip()}
    
    for ln in open(data_path, encoding='utf-8-sig'):
        if not ln.strip(): continue
        s = json.loads(ln)
        if s.get('source') != 'kg-qa-generated': continue
        
        s['fused_candidates_list'] = fused_map.get(s['id'], [])
        samples.append(s)
    return samples

# --- Metric Functions ---
def get_recall_at_k(cands, gold, k): return 1 if gold in {c['qid'] for c in (cands or [])[:k]} else 0
def get_mrr_at_k(cands, gold, k, is_qid_list=False):
    try:
        cand_list = cands[:k] if is_qid_list else [c['qid'] for c in (cands or [])[:k]]
        return 1.0 / (cand_list.index(gold) + 1)
    except ValueError: return 0.0
def get_ndcg_at_k(cands, gold, k, is_qid_list=False):
    try:
        cand_list = cands[:k] if is_qid_list else [c['qid'] for c in (cands or [])[:k]]
        return 1.0 / np.log2(cand_list.index(gold) + 2)
    except ValueError: return 0.0

# --- Statistical Test Functions ---
def bootstrap_ci(data, metric_func, n_bootstraps=N_BOOTSTRAP, **kwargs):
    """Calculates 95% bootstrap confidence interval for a given metric function."""
    rng = np.random.default_rng(42)
    bootstrap_scores = [metric_func(rng.choice(data, size=len(data), replace=True), **kwargs) for _ in range(n_bootstraps)]
    return np.percentile(bootstrap_scores, [2.5, 97.5])

def mcnemar_test(samples):
    """Performs McNemar test on fused vs. KG recall."""
    n01 = sum(1 for s in samples if s['fused_ok'] and not s['kg_ok'])
    n10 = sum(1 for s in samples if not s['fused_ok'] and s['kg_ok'])
    n = n01 + n10
    p_val = 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) * (0.5**n) if n > 0 else 1.0
    return {'n01': n01, 'n10': n10, 'p_value': p_val}

def approx_randomization_test(samples, n_trials=N_RANDOMIZATION):
    """Performs approximate randomization test for MRR@20 difference."""
    rng = np.random.default_rng(42)
    obs_diff = np.mean([s['mrr_fused'] - s['mrr_kg'] for s in samples])
    
    perm_diffs = []
    for _ in range(n_trials):
        shuffled_scores = [(s['mrr_fused'], s['mrr_kg']) if rng.random() < 0.5 else (s['mrr_kg'], s['mrr_fused']) for s in samples]
        perm_diff = np.mean([s1 - s2 for s1, s2 in shuffled_scores])
        perm_diffs.append(perm_diff)
        
    p_val = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return {'diff': obs_diff, 'p_value': p_val}

# --- Main Evaluation Logic ---
def main(args):
    samples = load_data(DATA_PATH, FUSED_PATH)
    
    # Pre-calculate correctness and scores for each sample
    for s in samples:
        gold = s.get('motif', [None]*5)[-1]
        s['gold'] = gold
        s['text_ok'] = get_recall_at_k(s.get('text_candidates'), gold, K)
        s['kg_ok'] = get_recall_at_k(s.get('kg_candidates'), gold, K)
        s['fused_ok'] = 1 if gold in s['fused_candidates_list'][:K] else 0
        s['oracle_ok'] = s['text_ok'] or s['kg_ok']
        s['mrr_fused'] = get_mrr_at_k(s['fused_candidates_list'], gold, K, is_qid_list=True)
        s['mrr_kg'] = get_mrr_at_k(s.get('kg_candidates'), gold, K)
        s['ndcg_fused'] = get_ndcg_at_k(s['fused_candidates_list'], gold, K, is_qid_list=True)
        s['ndcg_kg'] = get_ndcg_at_k(s.get('kg_candidates'), gold, K)

    # --- Calculate Metrics ---
    metrics = {}
    recall_metrics = ['text_ok', 'kg_ok', 'fused_ok', 'oracle_ok']
    for m in recall_metrics:
        mean_val = np.mean([s[m] for s in samples])
        ci = bootstrap_ci(samples, lambda x: np.mean([s[m] for s in x]))
        metrics[f'recall_{m.replace("_ok", "")}@20'] = {'mean': mean_val, 'ci95': ci.tolist()}

    ranking_metrics = ['mrr_fused', 'mrr_kg', 'ndcg_fused', 'ndcg_kg']
    for m in ranking_metrics:
        mean_val = np.mean([s[m] for s in samples])
        ci = bootstrap_ci(samples, lambda x: np.mean([s[m] for s in x]))
        metrics[m.replace('_fused', '@20_fused').replace('_kg', '@20_kg')] = {'mean': mean_val, 'ci95': ci.tolist()}

    # --- Calculate Derived & Statistical Metrics ---
    p_text = metrics['recall_text@20']['mean']
    p_kg = metrics['recall_kg@20']['mean']
    p_union = metrics['recall_oracle@20']['mean']
    metrics['independence_gap'] = p_union - (p_text + p_kg - p_text * p_kg)
    
    best_single_recall = max(p_text, p_kg)
    metrics['gap_closure'] = (metrics['recall_fused@20']['mean'] - best_single_recall) / max(1e-9, p_union - best_single_recall)

    num_gold_samples = len(samples)
    metrics['exact_ci'] = np.sum([1 for s in samples if s['fused_ok'] and not s['text_ok'] and not s['kg_ok']]) / max(1, num_gold_samples)
    
    metrics['mcnemar_overall'] = mcnemar_test(samples)
    metrics['mcnemar_by_slice'] = {s_name: mcnemar_test([s for s in samples if s.get('slice') == s_name]) for s_name in {'dense', 'bridge', 'kg'}}
    
    metrics['approx_randomization_mrr'] = approx_randomization_test(samples)

    # --- Output Results ---
    print(json.dumps(metrics, indent=2))
    if args.json_out:
        import os
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.json_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive evaluation harness for Phase 3.')
    parser.add_argument('--json_out', type=str, help='Path to save the output JSON file.')
    args = parser.parse_args()
    main(args)
