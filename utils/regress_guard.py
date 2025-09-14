import json, sys

# e.g., python utils/regress_guard.py data/phase3/annotated_12k_walked_b200_h4.jsonl 0.078 0.002
DATA=sys.argv[1]
BASE_KG=float(sys.argv[2])
TOL=float(sys.argv[3])

g=ok=0
for ln in open(DATA,encoding="utf-8-sig"):
    s=json.loads(ln)
    if s.get("source")!="kg-qa-generated": continue
    g+=1
    gold=(s.get("motif",[None]*5)[-1])
    if not gold:
        continue
    k=[c["qid"] for c in (s.get("kg_candidates") or [])[:20]]
    ok+=gold in k

kg=ok/max(1,g)
print(f"[guard] kg@20={kg:.3f} (baseline={BASE_KG:.3f}, tol={TOL:.3f})")
sys.exit(0 if kg+1e-12 >= BASE_KG - TOL else 2)
