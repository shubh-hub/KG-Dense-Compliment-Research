#!/usr/bin/env python3
"""
Run Entity Linking pipeline on a sample corpus.

Pipeline:
1. spaCy for Named Entity Recognition (NER)
2. Alias map for candidate generation
3. BM25 index for disambiguation
"""

import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any
import re

import spacy
from rank_bm25 import BM25Okapi
from rapidfuzz import process, fuzz
from tqdm import tqdm

def norm(s: str) -> str:
    return " ".join(s.lower().split())

# Entity hygiene filters
VALID_TYPES = {"PERSON","ORG","GPE","LOC","FAC","EVENT","WORK_OF_ART","PRODUCT","LANGUAGE","NORP"}
ORDINAL_RE = re.compile(r"^[0-9]+(st|nd|rd|th)$", re.I)

def good_mention(span):
    txt = span.text.strip()
    lbl = span.label_
    # type gate (skip DATE, TIME, ORDINAL, CARDINAL, QUANTITY, MONEY, PERCENT, etc.)
    if lbl not in VALID_TYPES:
        return False
    # hygiene gates
    if len(txt) < 2: return False
    if txt.isdigit(): return False
    if ORDINAL_RE.fullmatch(txt): return False
    # avoid 1-token lowercase common nouns like "congress", "season", etc.
    if len(txt.split()) == 1 and txt[0].islower():
        return False
    return True

def load_jsonl(file_path: str) -> List[Dict[Any, Any]]:
    """Load JSONL file."""
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items

def save_jsonl(items: List[Dict], output_path: str):
    """Save items to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def run_entity_linking(
    samples: List[Dict],
    alias_map: Dict,
    bm25_index: BM25Okapi,
    bm25_corpus: List[Dict],
    nlp,
    top_k_candidates: int = 5,
    disambiguation_threshold: float = 1.0
) -> List[Dict]:
    """Run the full entity linking pipeline on a list of samples."""
    
    # Build qid2idx for fast lookup
    qid2idx = {row["qid"]: i for i, row in enumerate(bm25_corpus)}
    alias_keys = set(alias_map.keys())
    
    processed_samples = []
    
    for sample in tqdm(samples, desc="Running Entity Linking"):
        text = sample.get('text', '') or sample.get('question', '')
        if not text:
            processed_samples.append(sample)
            continue

        doc = nlp(text)
        el_metadata = []

        for ent in doc.ents:
            if not good_mention(ent):
                continue
            mention = ent.text
            m = norm(mention)
            # Add alias normalization
            m = re.sub(r"^(the|a|an)\s+","", m)         # drop articles
            m = m.replace("'","'")                       # normalize apostrophes
            
            # 1. Candidate Generation: Exact-then-fuzzy
            candidate_qids = []
            
            # Fast exact path
            if m in alias_map:
                candidate_qids = alias_map[m]  # list of [qid, freq]
            else:
                # Cheap prefilter: only compare with aliases starting with same first char
                first = m[:1] if m else ""
                pool = [a for a in alias_keys if a.startswith(first)]
                # Fuzzy with cutoff to avoid scanning everything
                hits = process.extract(m, pool, limit=top_k_candidates, scorer=fuzz.WRatio, score_cutoff=75)
                for alias, score, _ in hits:
                    for qid, freq in alias_map[alias]:
                        candidate_qids.append((qid, freq))
            
            if not candidate_qids:
                el_metadata.append({"mention": mention, "qid": "NIL", "score": 0.0, "candidates": []})
                continue

            # De-dup candidates
            cand = {}
            for qid, freq in candidate_qids:
                cand[qid] = max(cand.get(qid, 0), freq)
            candidate_qids = list(cand.items())

            # 2. Disambiguation using BM25 with margin requirement
            # Include mention + ~30 tokens of local context
            loc = text.lower().find(mention.lower())
            window = 120
            left = max(0, loc - window); right = min(len(text), loc + len(mention) + window)
            query = (mention + " " + text[left:right]).strip()
            tokenized_query = query.split()

            # Get BM25 scores for candidate QIDs
            doc_scores = bm25_index.get_scores(tokenized_query)
            score_list = []
            for qid, _ in candidate_qids:
                idx = qid2idx.get(qid)
                if idx is not None:
                    score_list.append((qid, doc_scores[idx]))

            if not score_list:
                final_qid, best_score = "NIL", 0.0
            else:
                score_list.sort(key=lambda x: x[1], reverse=True)
                best_qid, best_score = score_list[0]
                second = score_list[1][1] if len(score_list) > 1 else -1e9
                margin = best_score - second
                # tighten threshold + margin
                if best_score < 1.5 or margin < 0.5:
                    final_qid = "NIL"
                else:
                    final_qid = best_qid

            el_metadata.append({
                "mention": mention,
                "qid": final_qid,
                "score": best_score,
                "candidates": [c[0] for c in candidate_qids]
            })

        sample['el_metadata'] = el_metadata
        processed_samples.append(sample)
        
    return processed_samples

def main():
    parser = argparse.ArgumentParser(description="Run Entity Linking on sample corpus")
    parser.add_argument("--corpus_file", required=True, help="Path to the combined sample corpus file (JSONL)")
    parser.add_argument("--alias_map", default="data/phase2/alias_map.json", help="Path to alias map")
    parser.add_argument("--bm25_index", default="data/phase2/bm25_index.pkl", help="Path to BM25 index")
    parser.add_argument("--bm25_corpus", default="data/phase2/entity_corpus.jsonl", help="Path to BM25 corpus")
    parser.add_argument("--out_file", required=True, help="Output file for annotated corpus")
    parser.add_argument("--spacy_model", default="en_core_web_sm", help="spaCy model to use")
    args = parser.parse_args()

    print("Loading resources...")
    samples = load_jsonl(args.corpus_file)
    with open(args.alias_map, 'r') as f:
        alias_map = json.load(f)
    with open(args.bm25_index, 'rb') as f:
        bm25_data = pickle.load(f)
    bm25_index = bm25_data['bm25']
    bm25_corpus = load_jsonl(args.bm25_corpus)
    
    print(f"Loading spaCy model: {args.spacy_model}")
    try:
        nlp = spacy.load(args.spacy_model, disable=["tagger","parser","lemmatizer","attribute_ruler"])
    except OSError:
        print(f"spaCy model '{args.spacy_model}' not found. Downloading...")
        spacy.cli.download(args.spacy_model)
        nlp = spacy.load(args.spacy_model, disable=["tagger","parser","lemmatizer","attribute_ruler"])

    annotated_samples = run_entity_linking(samples, alias_map, bm25_index, bm25_corpus, nlp)
    
    save_jsonl(annotated_samples, args.out_file)
    print(f"\nEntity linking complete. Saved {len(annotated_samples)} annotated samples to {args.out_file}")

if __name__ == "__main__":
    main()
