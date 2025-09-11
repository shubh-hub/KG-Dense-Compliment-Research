#!/usr/bin/env python3
import json
from pathlib import Path
import re
def norm(s): return re.sub(r"\s+"," ", " ".join(s.lower().split()))
U = set(x.strip() for x in open("data/phase2/U_qids_delta.txt") if x.strip())
out = {}
for ln in open("data/raw/kg/wikidata5m/wikidata5m_entity.txt", encoding="utf-8"):
    parts = ln.rstrip("\n").split("\t")
    if len(parts) < 2: continue
    qid, label = parts[0], parts[1]
    if qid in U and label:
        out.setdefault(norm(label), []).append([qid, 1])
Path("data/phase2").mkdir(parents=True, exist_ok=True)
json.dump(out, open("data/phase2/alias_map_delta.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"aliases (delta): {len(out):,}")
