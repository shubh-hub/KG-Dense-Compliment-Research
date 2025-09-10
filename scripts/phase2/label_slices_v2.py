#!/usr/bin/env python3
import json, argparse
from pathlib import Path
from statistics import mean

def load_jsonl(p):
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def save_jsonl(items, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def two_hop_reachable(qids, graph):
    """Check if any pair of QIDs is reachable within 2 hops"""
    if len(qids) < 2:
        return False
    
    qid_set = set(qids)
    for qid in qids:
        if qid in graph:
            # 1-hop neighbors
            neighbors = set(graph[qid])
            # Check if any other QID is a direct neighbor
            if neighbors & qid_set:
                return True
            # 2-hop neighbors
            for neighbor in neighbors:
                if neighbor in graph:
                    two_hop = set(graph[neighbor])
                    if two_hop & qid_set:
                        return True
    return False

def label_sample(s, graph):
    el = s.get("el_metadata", [])
    qid_scores = [(m.get("qid"), m.get("score", 0.0)) for m in el if m.get("qid") and m["qid"] != "NIL"]
    high_conf_qids = [qid for qid, score in qid_scores if score > 2.0]  # Confident links
    all_qids = [qid for qid, _ in qid_scores]
    
    any_qid = len(qid_scores) > 0
    nil_rate = 1.0 - (len(qid_scores) / max(1, len(el))) if el else 1.0

    # Router features for later analysis
    s["router_feats"] = {
        "n_mentions": len(el),
        "n_linked_qids": len(qid_scores),
        "n_conf_qids": len(high_conf_qids),
        "avg_qid_score": (mean([score for _, score in qid_scores]) if qid_scores else 0.0),
        "nil_rate": nil_rate,
        "two_hop_reachable": two_hop_reachable(all_qids, graph)
    }

    # Improved slice assignment heuristics:
    # KG slice: has ≥2 confident entities AND 2-hop reachable in KG
    if len(high_conf_qids) >= 2 and two_hop_reachable(high_conf_qids, graph):
        return "kg"
    # Dense slice: EL mostly NIL / low confidence (text-heavy)
    if nil_rate > 0.8 or not any_qid:
        return "dense"
    # Bridge: some EL signal but needs text reasoning too
    if any_qid and nil_rate <= 0.8:
        return "bridge"
    # Default fallback
    return "none"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated_corpus", required=True)
    ap.add_argument("--kg_graph", required=True, help="KG adjacency graph JSON")
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    print(f"[label_slices_v2] Loading KG graph from {args.kg_graph}")
    with open(args.kg_graph) as f:
        graph = json.load(f)
    print(f"[label_slices_v2] Loaded graph with {len(graph):,} nodes")

    print(f"[label_slices_v2] Processing {args.annotated_corpus}")
    
    out = []
    slice_counts = {"kg": 0, "dense": 0, "bridge": 0, "none": 0}
    
    for s in load_jsonl(args.annotated_corpus):
        slice_label = label_sample(s, graph)
        s["slice"] = slice_label
        slice_counts[slice_label] += 1
        out.append(s)
    
    save_jsonl(out, args.out_file)
    
    total = sum(slice_counts.values())
    print(f"[label_slices_v2] Processed {total:,} samples")
    print(f"[label_slices_v2] Slice distribution:")
    for slice_name, count in slice_counts.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {slice_name}: {count:,} ({pct:.1f}%)")
    
    print(f"[label_slices_v2] Saved to {args.out_file}")

if __name__ == "__main__":
    main()
