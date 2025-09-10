#!/usr/bin/env python3
import json
import statistics as st
from pathlib import Path

def main():
    p = Path("data/phase2/annotated_12k_v5.jsonl")
    if not p.exists():
        print(f"ERROR: Input file not found at {p}")
        return

    N = 0
    docs_EL = 0
    mentions = 0
    linked = 0
    margins = []
    topqids = set()

    with open(p, encoding="utf-8") as f:
        for ln in f:
            x = json.loads(ln)
            N += 1
            el = x.get("el_metadata", [])
            if el:
                docs_EL += 1
            for m in el:
                mentions += 1
                if m.get("qid") and m["qid"] != "NIL":
                    linked += 1
                    topqids.add(m["qid"])
                    margins.append(m.get("score", 0.0))

    print(f"Docs: {N} | with EL: {docs_EL} ({docs_EL/N:.1%})")
    print(f"Mentions: {mentions} | linked: {linked} ({linked/max(1,mentions):.1%})")
    print(f"Unique QIDs linked: {len(topqids)}")
    if margins:
        print(f"BM25 top-score stats (linked only): mean={st.mean(margins):.2f}, p50={st.median(margins):.2f}")
    else:
        print("No linked mentions found to calculate score stats.")

if __name__ == "__main__":
    main()
