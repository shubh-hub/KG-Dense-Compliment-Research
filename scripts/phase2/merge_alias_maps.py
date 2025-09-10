#!/usr/bin/env python3
import json
import collections
import os
from pathlib import Path

def main():
    alias_U_path = Path("data/phase2/alias_map_U.json")
    alias_filtered_path = Path("data/phase2/alias_map_filtered.json")
    output_path = Path("data/phase2/alias_map_merged.json")

    if not alias_U_path.exists():
        print(f"[merge_alias_maps] ERROR: Input file not found at {alias_U_path}")
        exit(1)

    print(f"[merge_alias_maps] Loading base alias map from {alias_U_path}")
    aU = json.load(open(alias_U_path))
    merged = collections.defaultdict(lambda: collections.Counter())
    for k, pairs in aU.items():
        for q, c in pairs: 
            merged[k][q] += c
    print(f"[merge_alias_maps] Loaded {len(aU):,} aliases from base map.")

    if os.path.exists(alias_filtered_path):
        print(f"[merge_alias_maps] Loading filtered alias map from {alias_filtered_path} to merge.")
        aS = json.load(open(alias_filtered_path))
        for k, pairs in aS.items():
            for q, c in pairs: 
                merged[k][q] += c
        print(f"[merge_alias_maps] Merged {len(aS):,} aliases from filtered map.")
    else:
        print(f"[merge_alias_maps] Filtered alias map not found at {alias_filtered_path}, skipping merge.")

    final = {k: sorted(list(v.items()), key=lambda x: -x[1]) for k, v in merged.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"[merge_alias_maps] Successfully created {output_path} with {len(final):,} total aliases.")

if __name__ == "__main__":
    main()
