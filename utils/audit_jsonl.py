#!/usr/bin/env python3
import json, sys
f = sys.argv[1]
tot=gold=gt=gk=gb=0
with open(f, encoding="utf-8-sig") as fh:
    for ln in fh:
        if not ln.strip(): continue
        s=json.loads(ln)
        tot+=1
        if s.get("source")=="kg-qa-generated":
            gold+=1
            if s.get("text_candidates"): gt+=1
            if s.get("kg_candidates"):   gk+=1
            if s.get("text_candidates") and s.get("kg_candidates"): gb+=1
print(f"total={tot} gold={gold} gold_with_text={gt} gold_with_kg={gk} gold_with_both={gb}")
if gold and gk/gold < 0.6:
    print("[WARN] gold KG coverage <60%")
