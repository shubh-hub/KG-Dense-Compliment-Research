import random, numpy as np, json

def mrr_at_k(ids, lists, golds, k=20):
    mrr=0;
    for i in ids:
        cand=lists[i][:k]; gold=golds[i]
        try: r=cand.index(gold)+1; mrr += 1.0/r
        except ValueError: pass
    return mrr/len(ids) if ids else 0.0

DATA="data/phase3/annotated_12k_walked_b200_h4.jsonl"
FUSED="data/phase3/annotated_12k_fused_b200_h4.jsonl"
fused, kg, golds, ids = {}, {}, {}, []
for ln in open(DATA,encoding="utf-8-sig"):
    s=json.loads(ln)
    if s.get("source")!="kg-qa-generated": continue
    ids.append(s["id"]); golds[s["id"]] = (s.get("motif",[None]*5)[-1])
    kg[s["id"]] = [c["qid"] for c in (s.get("kg_candidates") or [])]
for ln in open(FUSED,encoding="utf-8-sig"):
    s=json.loads(ln); fused[s["id"]] = [c["qid"] for c in (s.get("fused_candidates") or [])]

m_f = mrr_at_k(ids,fused,golds); m_k = mrr_at_k(ids,kg,golds)
obs = m_f - m_k

# Approximate randomization
np.random.seed(42)
diffs=[]
for _ in range(10000):
    mix={}
    for i in ids:
        # For each sample, randomly choose between the fused and kg list
        # This simulates the null hypothesis that they are interchangeable
        if np.random.rand() < 0.5:
            mix[i] = fused[i]
        else:
            mix[i] = kg[i]
    # The test statistic is the difference in MRR between the mixed set and the KG baseline
    # Under the null, this difference should be centered around 0
    diffs.append(mrr_at_k(ids,mix,golds) - m_k)

# The p-value is the proportion of simulated differences as extreme as the observed one
p = np.mean(np.abs(diffs) >= abs(obs))
print("MRR@20 Δ=", obs, "p≈", float(p))
