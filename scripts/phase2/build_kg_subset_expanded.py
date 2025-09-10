#!/usr/bin/env python3
import json, argparse, gzip
from pathlib import Path
from collections import defaultdict

def open_any(p):
    return gzip.open(p, "rt", encoding="utf-8", errors="ignore") if str(p).endswith(".gz") else open(p, "r", encoding="utf-8", errors="ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linked_qids", required=True)
    ap.add_argument("--wikidata_triples", required=True)
    ap.add_argument("--out_triples", required=True)
    ap.add_argument("--out_graph", required=True)
    args = ap.parse_args()

    L = {ln.strip() for ln in open(args.linked_qids) if ln.strip()}
    print(f"[kg_subset] linked: {len(L):,}")

    # Pass 1: collect 1-hop neighbors of linked QIDs; keep any triple touching L
    N1 = set()
    kept_1 = []
    processed = 0
    with open_any(args.wikidata_triples) as f:
        for line in f:
            processed += 1
            if processed % 1000000 == 0:
                print(f"[kg_subset] pass1: processed {processed:,}, kept {len(kept_1):,}, neighbors {len(N1):,}")
            
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            h, r, t = parts[0], parts[1], parts[2]
            if h in L or t in L:
                kept_1.append((h,r,t))
                if h in L: N1.add(t)
                if t in L: N1.add(h)

    U = L | N1
    print(f"[kg_subset] neighbors: {len(N1):,} | union U: {len(U):,} | pass1 kept: {len(kept_1):,}")

    # Pass 2: keep *all* triples whose endpoints are both in U (for connectivity among neighbors)
    kept_all = []
    processed = 0
    with open_any(args.wikidata_triples) as f:
        for line in f:
            processed += 1
            if processed % 1000000 == 0:
                print(f"[kg_subset] pass2: processed {processed:,}, kept {len(kept_all):,}")
            
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            h, r, t = parts[0], parts[1], parts[2]
            if h in U and t in U:
                kept_all.append((h,r,t))

    print(f"[kg_subset] pass2 kept (both in U): {len(kept_all):,}")

    # Build adjacency
    G = defaultdict(set)
    for h,r,t in kept_all:
        G[h].add(t); G[t].add(h)

    # Save
    Path(args.out_triples).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_triples,"w",encoding="utf-8") as fo:
        for h,r,t in kept_all:
            fo.write(json.dumps({"head":h,"relation":r,"tail":t}, ensure_ascii=False)+"\n")
    json.dump({k:list(v) for k,v in G.items()}, open(args.out_graph,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[kg_subset] nodes: {len(G):,} | edges: {sum(len(v) for v in G.values())//2:,}")

if __name__ == "__main__":
    main()
