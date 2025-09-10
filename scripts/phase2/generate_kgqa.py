#!/usr/bin/env python3
"""
Generate 1k KG-QA examples with 2-3 hop motifs from Wikidata5M.
"""

import json
import random
import argparse
import gzip
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List

def load_relations(file_path: str) -> Dict[str, str]:
    """Load relation labels from file."""
    relations = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                relations[parts[0]] = parts[1]
    return relations

def load_entities(file_path: str) -> Dict[str, str]:
    """Load entity labels from file."""
    entities = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entity in data:
            entities[entity['entity_id']] = entity['label']
    return entities

def generate_questions(triples_file: str, relations: Dict, entities: Dict, n_samples: int) -> List[Dict]:
    """Generate 2-hop questions from KG triples."""
    print("Building graph from triples...")
    graph = defaultdict(list)
    with gzip.open(triples_file, 'rt', errors='ignore') as f:
        for i, line in enumerate(f):
            if i > 500000: # Limit graph size for performance
                break
            parts = line.strip().split('\t')
            if len(parts) == 3:
                h, r, t = parts
                graph[h].append((r, t))

    print("Generating questions...")
    qa_pairs = []
    entity_ids = list(graph.keys())

    with tqdm(total=n_samples, desc="Generating KG-QA") as pbar:
        while len(qa_pairs) < n_samples:
            if not entity_ids:
                break
            
            # Pick a random starting entity
            start_entity_id = random.choice(entity_ids)
            if start_entity_id not in graph or start_entity_id not in entities:
                continue

            # Find a 2-hop path
            if len(graph[start_entity_id]) > 0:
                r1, mid_entity_id = random.choice(graph[start_entity_id])
                if mid_entity_id in graph and mid_entity_id in entities and len(graph[mid_entity_id]) > 0:
                    r2, end_entity_id = random.choice(graph[mid_entity_id])
                    if end_entity_id in entities:
                        # We have a path: start -> r1 -> mid -> r2 -> end
                        start_label = entities[start_entity_id]
                        mid_label = entities[mid_entity_id]
                        end_label = entities[end_entity_id]
                        r1_label = relations.get(r1, 'related to')
                        r2_label = relations.get(r2, 'associated with')

                        # Generate question and answer
                        question = f"What is the {r2_label} of the {r1_label} of {start_label}?"
                        answer = end_label

                        qa_pairs.append({
                            "id": f"kgqa_{len(qa_pairs)}",
                            "question": question,
                            "answer": answer,
                            "source": "kg-qa-generated",
                            "motif": [start_entity_id, r1, mid_entity_id, r2, end_entity_id]
                        })
                        pbar.update(1)

    return qa_pairs

def main():
    parser = argparse.ArgumentParser(description="Generate KG-QA examples")
    parser.add_argument("--triples", default="data/raw/kg/wikidata5m/triples_train.txt.gz", help="Path to training triples")
    parser.add_argument("--relations", default="data/raw/kg/wikidata5m/wikidata5m_relation.txt", help="Path to relations file")
    parser.add_argument("--entities", default="data/processed/kg/entities_clean.json", help="Path to clean entities file")
    parser.add_argument("--out_file", required=True, help="Output JSONL file path")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of KG-QA pairs to generate")
    args = parser.parse_args()

    # Load data
    relations = load_relations(args.relations)
    entities = load_entities(args.entities)

    # Generate questions
    qa_pairs = generate_questions(args.triples, relations, entities, args.n_samples)

    # Save to file
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"Generated {len(qa_pairs)} KG-QA examples and saved to {args.out_file}")

if __name__ == "__main__":
    main()
