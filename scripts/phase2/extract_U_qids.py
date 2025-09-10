#!/usr/bin/env python3
import json
from pathlib import Path

print("[extract_U_qids] Starting extraction...")

graph_path = Path("data/phase2/kg_subset_graph_expanded.json")
output_path = Path("data/phase2/U_qids.txt")

if not graph_path.exists():
    print(f"[extract_U_qids] ERROR: Input graph not found at {graph_path}")
    exit(1)

G = json.load(open(graph_path))
U = set(G.keys())
for v in G.values():
    U.update(v)

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    for q in sorted(U):
        f.write(q + "\n")

print(f"[extract_U_qids] Successfully created {output_path} with {len(U):,} QIDs.")
