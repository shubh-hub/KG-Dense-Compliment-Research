import json, numpy as np
DATA="data/phase3/annotated_12k_walked_final_v6_textfix_r2.jsonl"
FUSED="data/phase3/annotated_12k_fused_lr5bin_v6_textfix_r2_final.jsonl"

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

fused_map={}
for ln in open(FUSED,encoding="utf-8-sig"):
    s=json.loads(ln); fused_map[s["id"]]=[c["qid"] for c in (s.get("fused_candidates") or [])]
ids=[]; golds={}; kg_map={}
for ln in open(DATA,encoding="utf-8-sig"):
    s=json.loads(ln)
    if s.get("source")!="kg-qa-generated": continue
    ids.append(s["id"]); golds[s["id"]]=(s.get("motif",[None]*5)[-1])
    kg_map[s["id"]]=[c["qid"] for c in (s.get("kg_candidates") or [])]

print("MRR@20 fused:", mrr_at_k(ids,fused_map,golds), "kg:", mrr_at_k(ids,kg_map,golds))
print("NDCG@20 fused:", ndcg_at_k(ids,fused_map,golds), "kg:", ndcg_at_k(ids,kg_map,golds))
