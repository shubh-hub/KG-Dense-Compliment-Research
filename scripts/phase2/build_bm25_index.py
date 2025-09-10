#!/usr/bin/env python3
"""
Build BM25 index over entity abstracts for disambiguation.
Uses entity labels + descriptions as "abstracts" for ranking.
"""

import json
import re
import argparse
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def normalize_text(text):
    """Normalize text for BM25 indexing."""
    if not text:
        return ""
    # Convert to lowercase, remove special chars, normalize whitespace
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r'\s+', ' ', text.strip())

def main():
    parser = argparse.ArgumentParser(description="Build BM25 index for entity disambiguation")
    parser.add_argument("--entities", required=True, help="Path to processed entities.json")
    parser.add_argument("--out_corpus", required=True, help="Output corpus jsonl path")
    parser.add_argument("--out_bm25", required=True, help="Output BM25 pickle path")
    parser.add_argument("--max_entities", type=int, default=None, help="Limit entities for testing")
    args = parser.parse_args()

    docs = []
    qids = []
    processed = 0

    print(f"Building entity corpus from {args.entities}...")
    
    # Create output directories
    Path(args.out_corpus).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_bm25).parent.mkdir(parents=True, exist_ok=True)

    # Load JSON array format
    with open(args.entities, 'r', encoding='utf-8') as f:
        entities_data = json.load(f)
    
    print(f"Loaded {len(entities_data)} entities from file")
    
    with open(args.out_corpus, 'w', encoding='utf-8') as fout:
        
        for entity in tqdm(entities_data, desc="Processing entities"):
            try:
                qid = entity.get('entity_id') or entity.get('id') or entity.get('qid')
                
                if not qid or not qid.startswith('Q'):
                    continue
                
                # Combine label and description as "abstract"
                label = entity.get('label', '')
                description = entity.get('description', '') or entity.get('desc', '')
                
                # Create abstract text
                abstract_parts = []
                if label:
                    abstract_parts.append(label)
                if description:
                    abstract_parts.append(description)
                
                abstract_text = ' '.join(abstract_parts).strip()
                if not abstract_text:
                    abstract_text = qid  # Fallback to QID
                
                # Save to corpus file
                corpus_entry = {
                    "qid": qid,
                    "text": abstract_text
                }
                fout.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')
                
                # Prepare for BM25
                normalized_text = normalize_text(abstract_text)
                docs.append(normalized_text.split())
                qids.append(qid)
                
                processed += 1
                if args.max_entities and processed >= args.max_entities:
                    break
                    
            except Exception as e:
                print(f"Error processing entity {entity.get('entity_id', 'unknown')}: {e}")
                continue

    print(f"Processed {processed:,} entities")
    print(f"Building BM25 index...")

    # Build BM25 index
    bm25 = BM25Okapi(docs)
    
    # Save BM25 index and metadata
    # Build qid2idx for fast lookup later
    qid2idx = {qid: i for i, qid in enumerate(qids)}
    bm25_data = {
        'bm25': bm25,
        'qids': qids,
        'qid2idx': qid2idx,
        'num_docs': len(docs)
    }
    
    with open(args.out_bm25, 'wb') as f:
        pickle.dump(bm25_data, f)
    
    print(f"BM25 index saved to {args.out_bm25}")
    print(f"Corpus saved to {args.out_corpus}")
    print(f"Index contains {len(qids):,} entities | qid2idx entries: {len(qid2idx):,}")

if __name__ == "__main__":
    main()
