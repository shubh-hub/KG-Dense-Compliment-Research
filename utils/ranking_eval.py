import json, numpy as np

def mrr_at_k(ids, lists, golds, k=20):
    mrr=0; n=0
    for i in ids:
        gold=golds[i]; cand=lists[i][:k]; n+=1
        try: r=cand.index(gold)+1; mrr+=1.0/r
        except ValueError: pass
    return mrr/max(1,n)

def ndcg_at_k(ids, lists, golds, k=20):
    nd=0; n=0
    for i in ids:
        gold=golds[i]; cand=lists[i][:k]; n+=1
        try: r=cand.index(gold)+1; nd+=1/np.log2(r+1)
        except ValueError: pass
    return nd/max(1,n)

def main(args):
    fused_map={}
    for ln in open(args.fused,encoding="utf-8-sig"):
        s=json.loads(ln); fused_map[s["id"]]=[c["qid"] for c in (s.get("fused_candidates") or [])]
    ids=[]; golds={}; kg_map={}
    for ln in open(args.walked,encoding="utf-8-sig"):
        s=json.loads(ln)
        if s.get("source")!="kg-qa-generated": continue
        ids.append(s["id"]); golds[s["id"]]=(s.get("motif",[None]*5)[-1])
        kg_map[s["id"]]=[c["qid"] for c in (s.get("kg_candidates") or [])]

    print("--- Ranking Metrics ---")
    print(f"| Metric  | Fused     | KG        |")
    print(f"|---------|-----------|-----------|")
    print(f"| MRR@20  | {mrr_at_k(ids,fused_map,golds):.5f}   | {mrr_at_k(ids,kg_map,golds):.5f}   |")
    print(f"| NDCG@20 | {ndcg_at_k(ids,fused_map,golds):.5f}   | {ndcg_at_k(ids,kg_map,golds):.5f}   |")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate ranking metrics.')
    parser.add_argument('--walked', type=str, required=True, help='Path to the walked JSONL file.')
    parser.add_argument('--fused', type=str, required=True, help='Path to the fused JSONL file.')
    args = parser.parse_args()
    main(args)
