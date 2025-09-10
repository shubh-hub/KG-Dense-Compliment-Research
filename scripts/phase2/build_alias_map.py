#!/usr/bin/env python3
"""
Build Wikidata5M alias map with normalized keys and QID priors.
Adapted for our existing processed data structure.
"""

import json
import re
import argparse
from collections import defaultdict
from pathlib import Path
import sys
from tqdm import tqdm

def norm(s: str) -> str:
    """Normalize alias string for matching."""
    return re.sub(r'\s+', ' ', s.strip().lower())

MAX_ALIAS_LEN = 80
MAX_QIDS_PER_ALIAS = 10

def main():
    parser = argparse.ArgumentParser(description="Build alias map from Wikidata5M entities")
    parser.add_argument("--entities", required=True, help="Path to processed entities.json")
    parser.add_argument("--out", required=True, help="Output alias map path")
    parser.add_argument("--max_entities", type=int, default=None, help="Limit entities for testing")
    args = parser.parse_args()

    alias2qids = defaultdict(lambda: defaultdict(int))
    processed = 0
    dropped_too_long = 0

    def emit_alias(label, qid):
        """Add alias to mapping with frequency count, with quality guardrails."""
        nonlocal dropped_too_long
        if not label or len(label.strip()) < 2:
            return
        if len(label.strip()) > MAX_ALIAS_LEN:
            dropped_too_long += 1
            return
        normalized = norm(label)
        if normalized:
            alias2qids[normalized][qid] += 1

    print(f"Processing entities from {args.entities}...")
    
    # Load JSON array format
    with open(args.entities, 'r', encoding='utf-8') as f:
        entities_data = json.load(f)
    
    print(f"Loaded {len(entities_data)} entities from file")
    
    for entity in tqdm(entities_data, desc="building alias map"):
        try:
            qid = entity.get('entity_id') or entity.get('id') or entity.get('qid')
            
            if not qid or not qid.startswith('Q'):
                continue
            
            # Add main label (true name)
            label = entity.get('label')
            if label:
                emit_alias(label, qid)

            # Add aliases
            aliases = entity.get('aliases', [])
            if isinstance(aliases, list):
                for alias in aliases:
                    emit_alias(alias, qid)
            
            processed += 1
            
            if args.max_entities and processed >= args.max_entities:
                break
                
        except Exception as e:
            print(f"Error processing entity {entity.get('entity_id', 'unknown')}: {e}")
            continue

    print(f"Processed {processed:,} entities total")
    print(f"Found {len(alias2qids):,} unique normalized aliases")
    print(f"Dropped aliases > {MAX_ALIAS_LEN} chars: {dropped_too_long:,}")

    # Convert to final format: {alias: [(qid, count), ...]} sorted by count
    final_map = {}
    for alias, qid_counts in alias2qids.items():
        sorted_qids = sorted(qid_counts.items(), key=lambda x: -x[1])[:MAX_QIDS_PER_ALIAS]
        final_map[alias] = sorted_qids

    # Save alias map
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_map, f, ensure_ascii=False, indent=2)
    
    print(f"Alias map saved to {output_path}")
    print(f"Map size: {len(final_map):,} aliases")
    
    # Sample statistics
    total_mappings = sum(len(qids) for qids in final_map.values())
    avg_qids_per_alias = total_mappings / len(final_map) if final_map else 0
    print(f"Total QID mappings: {total_mappings:,}")
    print(f"Average QIDs per alias: {avg_qids_per_alias:.2f}")

if __name__ == "__main__":
    main()
