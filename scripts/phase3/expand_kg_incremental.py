#!/usr/bin/env python3
import json
import gzip
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def open_any(p):
    return gzip.open(p, "rt", encoding="utf-8", errors="ignore") if p.endswith(".gz") else open(p, "r", encoding="utf-8", errors="ignore")

def main():
    base_graph_path = Path("data/phase2/kg_subset_graph_expanded.json")
    delta_qids_path = Path("data/phase2/U_qids_delta.txt")
    all_triples_path = "data/raw/kg/wikidata5m/wikidata5m_all_triplet.txt.gz"
    out_graph_path = Path("data/phase2/kg_subset_graph_expanded_v2.json")
    out_triples_path = Path("data/phase2/kg_subset_triples_expanded_v2.jsonl")

    print(f"[INFO] Loading base graph from {base_graph_path}")
    with open(base_graph_path, 'r', encoding='utf-8') as f:
        G = json.load(f)
    
    base_nodes = set(G.keys())
    for node, neighbors in G.items():
        base_nodes.update(neighbors)
    print(f"[INFO] Base graph loaded with {len(base_nodes):,} nodes.")

    print(f"[INFO] Loading delta QIDs from {delta_qids_path}")
    delta_qids = {line.strip() for line in open(delta_qids_path, 'r') if line.strip()}
    print(f"[INFO] Loaded {len(delta_qids):,} delta QIDs.")

    # Pass 1: Find 1-hop neighbors of delta QIDs
    print("[INFO] Pass 1: Finding 1-hop neighbors of delta QIDs...")
    neighbors_of_delta = set()
    with open_any(all_triples_path) as f:
        for line in tqdm(f, desc="Scanning triples for neighbors"):
            try:
                h, r, t = line.strip().split('\t')
                if h in delta_qids:
                    neighbors_of_delta.add(t)
                if t in delta_qids:
                    neighbors_of_delta.add(h)
            except ValueError:
                continue
    
    # Define the new, expanded universe of nodes
    U_new = base_nodes | delta_qids | neighbors_of_delta
    print(f"[INFO] Expanded universe size: {len(U_new):,} nodes.")

    # Pass 2: Rebuild graph and triples within the new universe
    print("[INFO] Pass 2: Building new graph and triples file...")
    new_G = defaultdict(list)
    new_triples_count = 0
    with open(out_triples_path, 'w', encoding='utf-8') as f_triples, open_any(all_triples_path) as f_all:
        for line in tqdm(f_all, desc="Extracting final triples"):
            try:
                h, r, t = line.strip().split('\t')
                if h in U_new and t in U_new:
                    new_G[h].append(t)
                    f_triples.write(json.dumps({"head": h, "relation": r, "tail": t}) + "\n")
                    new_triples_count += 1
            except ValueError:
                continue

    # Ensure symmetry for undirected graph representation
    final_G = defaultdict(list)
    for h, neighbors in new_G.items():
        for t in neighbors:
            final_G[h].append(t)
            final_G[t].append(h)
    
    # Deduplicate neighbor lists
    for node in final_G:
        final_G[node] = sorted(list(set(final_G[node])))

    print(f"[INFO] Writing final expanded graph to {out_graph_path}")
    with open(out_graph_path, 'w', encoding='utf-8') as f_graph:
        json.dump(final_G, f_graph)

    final_node_count = len(final_G)
    print(f"\n[SUCCESS] Incremental graph expansion complete.")
    print(f"  - New graph: {out_graph_path} ({final_node_count:,} nodes)")
    print(f"  - New triples: {out_triples_path} ({new_triples_count:,} triples)")

if __name__ == "__main__":
    main()
