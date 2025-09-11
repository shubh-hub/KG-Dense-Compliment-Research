#!/usr/bin/env python3
import json, sys
U = set(x.strip() for x in open("data/phase2/U_qids.txt") if x.strip())
need = set()
src = "data/phase2/annotated_12k_routed_kg_text.jsonl"  # has all 1k KG-QA
for ln in open(src, encoding="utf-8"):
    s = json.loads(ln)
    if s.get("source")!="kg-qa-generated": continue
    m = s.get("motif",[])
    if len(m) >= 5:
        # include start, mid, end to ensure connectivity
        for q in (m[0], m[2], m[4]):
            if q not in U: need.add(q)
open("data/phase2/U_qids_delta.txt","w").write("\n".join(sorted(need))+"\n")
print(f"delta QIDs: {len(need)}")
