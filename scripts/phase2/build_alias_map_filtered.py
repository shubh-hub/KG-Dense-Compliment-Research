#!/usr/bin/env python3
import argparse, json, gzip
from pathlib import Path

def norm(s): return " ".join(s.lower().split())
def open_any(p): 
    return gzip.open(p, "rt", encoding="utf-8", errors="ignore") if str(p).endswith(".gz") else open(p, "r", encoding="utf-8", errors="ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mentions", required=True, help="normalized mentions, one per line")
    ap.add_argument("--entity_labels", required=True, help="wikidata5m_entity.txt (QID<TAB>Label)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[build_alias_map_filtered] Loading mentions from {args.mentions}")
    # Load the small mention set
    mention_set = set(m.strip() for m in open(args.mentions, encoding="utf-8") if m.strip())
    print(f"[build_alias_map_filtered] Loaded {len(mention_set):,} unique mentions")
    
    alias2qids = {}  # {alias: {qid: freq}}; freq=1 here
    matched = 0
    processed = 0

    print(f"[build_alias_map_filtered] Processing entity labels from {args.entity_labels}")
    with open_any(args.entity_labels) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2: continue
            qid, label = parts[0].strip(), parts[1].strip()
            if not qid.startswith("Q") or not label: continue
            
            processed += 1
            if processed % 100000 == 0:
                print(f"[build_alias_map_filtered] Processed {processed:,} entities, matched {matched:,}")
            
            a = norm(label)
            if a in mention_set:
                matched += 1
                alias2qids.setdefault(a, {})
                alias2qids[a][qid] = alias2qids[a].get(qid, 0) + 1

    print(f"[build_alias_map_filtered] Final: processed {processed:,} entities, matched {matched:,}")

    # finalize to desired format {alias: [(qid, count), ...]}
    final = {alias: sorted(list(qids.items()), key=lambda x: -x[1]) for alias, qids in alias2qids.items()}
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(final, open(args.out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[build_alias_map_filtered] Saved {len(final):,} aliases to {args.out}")

if __name__ == "__main__":
    main()
