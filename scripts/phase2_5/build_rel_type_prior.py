import json
import os
from collections import defaultdict, Counter

# Define constants
TRIPLES_PATH = 'data/phase2/kg_subset_triples_expanded_v2.jsonl'
ENTITY_TYPES_PATH = 'data/phase2/type/entity_types.json'
OUTPUT_DIR = 'data/phase2/type'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'rel_type_prior.json')
TOP_K_TYPES = 10

def main():
    """Parses KG triples and entity types to create relation-level type priors."""
    print(f"Loading entity types from {ENTITY_TYPES_PATH}...")
    with open(ENTITY_TYPES_PATH, 'r', encoding='utf-8') as f:
        entity_types = json.load(f)

    print(f"Reading triples from {TRIPLES_PATH}...")
    rel_type_info = defaultdict(lambda: {'head_types': Counter(), 'tail_types': Counter()})

    with open(TRIPLES_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                head, rel, tail = data.get('head'), data.get('relation'), data.get('tail')

                if not (head and rel and tail):
                    continue

                if head in entity_types:
                    for t in entity_types[head]:
                        rel_type_info[rel]['head_types'][t] += 1
                
                if tail in entity_types:
                    for t in entity_types[tail]:
                        rel_type_info[rel]['tail_types'][t] += 1

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Skipping malformed line {i+1}: {e}")

    # Consolidate and keep only top-K types
    final_priors = {}
    for rel, data in rel_type_info.items():
        final_priors[rel] = {
            'head_types': {t: c for t, c in data['head_types'].most_common(TOP_K_TYPES)},
            'tail_types': {t: c for t, c in data['tail_types'].most_common(TOP_K_TYPES)}
        }

    print(f"Generated priors for {len(final_priors)} relations.")
    print(f"Saving relation type priors to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_priors, f, indent=2)

    print("Done.")

if __name__ == '__main__':
    main()
