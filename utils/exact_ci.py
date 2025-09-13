import json
DATA="data/phase3/annotated_12k_walked_final_v6_textfix_r2.jsonl"
FUSED="data/phase3/annotated_12k_fused_lr5bin_v6_textfix_r2_final.jsonl"

fused_map={}
for ln in open(FUSED,encoding="utf-8-sig"):
    s=json.loads(ln)
    fused_map[s["id"]] = {c["qid"] for c in (s.get("fused_candidates") or [])[:20]}

g=ci=0
for ln in open(DATA,encoding="utf-8-sig"):
    if not ln.strip(): continue
    s=json.loads(ln)
    if s.get("source")!="kg-qa-generated": continue
    g+=1
    gold=(s.get("motif",[None]*5)[-1])
    t_ok = any(c["qid"]==gold for c in (s.get("text_candidates") or [])[:20])
    k_ok = any(c["qid"]==gold for c in (s.get("kg_candidates")   or [])[:20])
    f_ok = gold in fused_map.get(s["id"], set())
    if f_ok and (not t_ok) and (not k_ok):
        ci += 1
print(f"Exact CI = {ci/g:.4f}")
