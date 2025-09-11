#!/usr/bin/env python3
import json, argparse

def gold_qid(s):
    return s.get("motif",[None]*5)[-1] if s.get("source")=="kg-qa-generated" else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    args = ap.parse_args()
    
    samples = [json.loads(ln) for ln in open(args.in_jsonl, encoding="utf-8")]
    
    # Split by entity linking status
    linked_samples = []
    unlinked_samples = []
    
    for s in samples:
        g = gold_qid(s)
        if not g: continue
        
        # Check if any entity is linked (score > 0)
        has_linked = any(m.get("score", 0) > 0 for m in s.get("el_metadata", []))
        
        if has_linked:
            linked_samples.append(s)
        else:
            unlinked_samples.append(s)
    
    def evaluate_slice(samples_slice, slice_name):
        if not samples_slice:
            print(f"{slice_name}: No samples")
            return
            
        hits = {1: 0, 5: 0, 10: 0, 20: 0}
        total = len(samples_slice)
        
        for s in samples_slice:
            g = gold_qid(s)
            fused_candidates = s.get("fused_candidates", [])
            
            if fused_candidates:
                ranked_qids = [c["qid"] for c in fused_candidates]
                for k in [1, 5, 10, 20]:
                    if g in ranked_qids[:k]:
                        hits[k] += 1
        
        recall = {k: hits[k] / total for k in hits}
        print(f"{slice_name} (N={total}): @1={recall[1]:.3f}, @5={recall[5]:.3f}, @10={recall[10]:.3f}, @20={recall[20]:.3f}")
    
    evaluate_slice(linked_samples, "Linked entities")
    evaluate_slice(unlinked_samples, "Unlinked entities")

if __name__ == "__main__":
    main()
