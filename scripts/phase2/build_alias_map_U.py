#!/usr/bin/env python3
import json
import re
from pathlib import Path

def norm(s):
    return re.sub(r"\s+", " ", " ".join(s.lower().split()))

def main():
    u_qids_path = Path("data/phase2/U_qids.txt")
    entity_labels_path = Path("data/raw/kg/wikidata5m/wikidata5m_entity.txt")
    output_path = Path("data/phase2/alias_map_U.json")

    print(f"[build_alias_map_U] Loading QIDs from {u_qids_path}")
    U = {ln.strip() for ln in open(u_qids_path) if ln.strip()}
    print(f"[build_alias_map_U] Loaded {len(U):,} QIDs.")

    alias = {}
    print(f"[build_alias_map_U] Processing labels from {entity_labels_path}")
    with open(entity_labels_path, encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 2: continue
            qid, label = parts[0], parts[1]
            if qid in U and label:
                key = norm(label)
                alias.setdefault(key, [])
                alias[key].append([qid, 1])
            if i % 2_000_000 == 0:
                print(f"[build_alias_map_U] Scanned {i:,} labels...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alias, f, ensure_ascii=False, indent=2)
    
    print(f"[build_alias_map_U] Successfully created {output_path} with {len(alias):,} aliases.")

if __name__ == "__main__":
    main()
