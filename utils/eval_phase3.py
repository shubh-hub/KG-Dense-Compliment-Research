import json

DATA="data/phase3/annotated_12k_walked_final_v6_textfix_r2.jsonl"
FUSED="data/phase3/annotated_12k_fused_lr5bin_v6_textfix_r2_final.jsonl"

def topk_has(cands, qid, k=20):
    return any(x["qid"]==qid for x in (cands or [])[:k])

# load fused scores by id
fused_map={}
for ln in open(FUSED,encoding="utf-8-sig"):
    s=json.loads(ln)
    fused_map[s["id"]]=[c["qid"] for c in (s.get("fused_candidates") or [])[:20]]

n=g=text_ok=kg_ok=fused_ok=oracle_ok=0
for ln in open(DATA,encoding="utf-8-sig"):
    if not ln.strip(): continue
    s=json.loads(ln); n+=1
    if s.get("source")!="kg-qa-generated": continue
    g+=1
    gold = (s.get("motif",[None]*5)[-1])
    t_ok = topk_has(s.get("text_candidates"), gold, 20); text_ok += t_ok
    k_ok = topk_has(s.get("kg_candidates"), gold, 20);   kg_ok   += k_ok
    f_ok = gold in set(fused_map.get(s["id"], []));      fused_ok+= f_ok
    oracle_ok += (t_ok or k_ok)

def pct(x, d): return 100.0*x/max(1,d)
print(f"gold={g}")
print(f"text@20={pct(text_ok,g):.1f}  kg@20={pct(kg_ok,g):.1f}  oracle@20={pct(oracle_ok,g):.1f}")
print(f"fused@20={pct(fused_ok,g):.1f}")

best_single = max(text_ok, kg_ok)
oracle = oracle_ok
fused = fused_ok
gap = oracle - best_single
closure = (fused - best_single)/max(1, gap) if gap>0 else 0.0
print(f"gap_closure={closure*100:.1f}%")

# Complementarity Index: P(hybrid correct & both singles wrong)
hybrid_rescue = fused_ok - max(text_ok, kg_ok)  # upper bound approx
ci = max(0.0, hybrid_rescue / g)
print(f"CI≈{ci*100:.1f}%")
