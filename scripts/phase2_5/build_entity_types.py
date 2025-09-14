import json
import os
from collections import defaultdict

# Define constants
TRIPLES_PATH = 'data/phase2/kg_subset_triples_expanded_v2.jsonl'
OUTPUT_DIR = 'data/phase2/type'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'entity_types.json')
INSTANCE_OF_RELATION = 'P31'

def main():
    """Parses the KG triples to extract a map of entity QIDs to their types (from P31 relations)."""
    print(f"Reading triples from {TRIPLES_PATH}...")
    entity_types = defaultdict(set)
    
    with open(TRIPLES_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                if data.get('relation') == INSTANCE_OF_RELATION:
                    head = data.get('head')
                    tail = data.get('tail')
                    if head and tail:
                        entity_types[head].add(tail)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Skipping malformed line {i+1}: {e}")

    # Convert sets to lists for JSON serialization
    serializable_entity_types = {k: list(v) for k, v in entity_types.items()}

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Found types for {len(serializable_entity_types)} entities.")
    print(f"Saving entity types to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(serializable_entity_types, f, indent=2)

    print("Done.")

if __name__ == '__main__':
    main()
