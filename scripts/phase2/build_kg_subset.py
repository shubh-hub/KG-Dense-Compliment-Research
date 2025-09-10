#!/usr/bin/env python3
import json, argparse, gzip
from pathlib import Path
from collections import defaultdict

def open_any(p):
    return gzip.open(p, "rt", encoding="utf-8", errors="ignore") if str(p).endswith(".gz") else open(p, "r", encoding="utf-8", errors="ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linked_qids", required=True, help="File with linked QIDs, one per line")
    ap.add_argument("--wikidata_triples", required=True, help="Wikidata5M triples file")
    ap.add_argument("--out_triples", required=True, help="Output filtered triples JSONL")
    ap.add_argument("--out_graph", required=True, help="Output adjacency graph JSON")
    args = ap.parse_args()

    print(f"[build_kg_subset] Loading linked QIDs from {args.linked_qids}")
    with open(args.linked_qids) as f:
        linked_qids = set(line.strip() for line in f if line.strip())
    print(f"[build_kg_subset] Loaded {len(linked_qids):,} linked QIDs")

    # Build adjacency graph and collect relevant triples
    graph = defaultdict(set)  # qid -> set of connected qids
    triples = []
    processed = 0
    kept = 0

    print(f"[build_kg_subset] Processing triples from {args.wikidata_triples}")
    with open_any(args.wikidata_triples) as f:
        for line in f:
            processed += 1
            if processed % 1000000 == 0:
                print(f"[build_kg_subset] Processed {processed:,} triples, kept {kept:,}")
            
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    # Keep triples where both head and tail are in our linked QIDs
                    if h in linked_qids and t in linked_qids:
                        graph[h].add(t)
                        graph[t].add(h)  # bidirectional
                        triples.append({"head": h, "relation": r, "tail": t})
                        kept += 1
            except Exception as e:
                continue

    print(f"[build_kg_subset] Final: processed {processed:,} triples, kept {kept:,}")
    print(f"[build_kg_subset] Graph has {len(graph):,} nodes")

    # Save filtered triples
    Path(args.out_triples).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_triples, 'w', encoding='utf-8') as f:
        for triple in triples:
            f.write(json.dumps(triple, ensure_ascii=False) + '\n')

    # Save adjacency graph (convert sets to lists for JSON)
    graph_json = {qid: list(neighbors) for qid, neighbors in graph.items()}
    with open(args.out_graph, 'w', encoding='utf-8') as f:
        json.dump(graph_json, f, ensure_ascii=False, indent=2)

    print(f"[build_kg_subset] Saved {len(triples):,} triples to {args.out_triples}")
    print(f"[build_kg_subset] Saved graph to {args.out_graph}")

if __name__ == "__main__":
    main()
