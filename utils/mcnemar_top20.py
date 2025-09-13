import json, math
DATA="data/phase3/annotated_12k_walked_final_v6_textfix_r2.jsonl"
FUSED="data/phase3/annotated_12k_fused_lr5bin_v6_textfix_r2_final.jsonl"

fused_map={}
for ln in open(FUSED,encoding="utf-8-sig"):
    s=json.loads(ln)
    fused_map[s["id"]]=set(c["qid"] for c in (s.get("fused_candidates") or [])[:20])

n01=n10=0  # baseline wrong/fused right, baseline right/fused wrong
for ln in open(DATA,encoding="utf-8-sig"):
    if not ln.strip(): continue
    s=json.loads(ln)
    if s.get("source")!="kg-qa-generated": continue
    gold=(s.get("motif",[None]*5)[-1])
    kg_ok=any(c["qid"]==gold for c in (s.get("kg_candidates") or [])[:20])
    fused_ok=gold in fused_map.get(s["id"], set())
    n01 += (not kg_ok) and fused_ok
    n10 += kg_ok and (not fused_ok)

# exact binomial McNemar p-value
n = n01 + n10
m = min(n01, n10)
from math import comb
p = 2*sum(comb(n, i) for i in range(0, m+1))*(0.5**n) if n>0 else 1.0
print(f"McNemar fused vs KG: n01={n01}, n10={n10}, p={p:.4g}")
