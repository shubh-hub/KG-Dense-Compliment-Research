#!/usr/bin/env python3
import argparse, json, gzip, re, pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

def open_any(p):
    return gzip.open(p, "rt", encoding="utf-8", errors="ignore") if str(p).endswith(".gz") else open(p, "r", encoding="utf-8", errors="ignore")

def normalize_text(text):
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+"," ", text.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias_map", required=True)
    ap.add_argument("--entity_desc", required=True, help="wikidata5m_text.txt.gz (QID<TAB>Description)")
    ap.add_argument("--entity_labels", required=True, help="wikidata5m_entity.txt to add label into BM25 text")
    ap.add_argument("--out_corpus", required=True)
    ap.add_argument("--out_bm25", required=True)
    args = ap.parse_args()

    print(f"[build_bm25_filtered] Loading alias map from {args.alias_map}")
    alias_map = json.load(open(args.alias_map, encoding="utf-8"))
    needed_qids = set()
    for pairs in alias_map.values():
        for qid, _cnt in pairs:
            needed_qids.add(qid)
    print(f"[build_bm25_filtered] Need BM25 entries for {len(needed_qids):,} QIDs")

    # Small map: qid -> label (for presentation/search); only for needed_qids
    print(f"[build_bm25_filtered] Loading labels from {args.entity_labels}")
    qid2label = {}
    processed_labels = 0
    with open_any(args.entity_labels) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2: continue
            qid, label = parts[0], parts[1]
            processed_labels += 1
            if processed_labels % 100000 == 0:
                print(f"[build_bm25_filtered] Processed {processed_labels:,} labels, found {len(qid2label):,} needed")
            if qid in needed_qids: 
                qid2label[qid] = label

    print(f"[build_bm25_filtered] Found labels for {len(qid2label):,}/{len(needed_qids):,} needed QIDs")

    docs_tokens, qids = [], []
    Path(args.out_corpus).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[build_bm25_filtered] Building corpus from {args.entity_desc}")
    processed_desc = 0
    found_desc = 0
    
    with open(args.out_corpus, "w", encoding="utf-8") as fout, open_any(args.entity_desc) as fin:
        for line in fin:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2: continue
            qid, desc = parts
            processed_desc += 1
            
            if processed_desc % 100000 == 0:
                print(f"[build_bm25_filtered] Processed {processed_desc:,} descriptions, found {found_desc:,} needed")
            
            if qid not in needed_qids: continue
            found_desc += 1
            
            label = qid2label.get(qid, "")
            text = (label + " " + desc).strip() if label else desc
            entry = {"qid": qid, "text": text}
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            qids.append(qid)
            docs_tokens.append(normalize_text(text).split())

    print(f"[build_bm25_filtered] Building BM25 index for {len(qids):,} entities")
    bm25 = BM25Okapi(docs_tokens)
    
    # Include qid2idx for fast lookup
    qid2idx = {qid: i for i, qid in enumerate(qids)}
    
    bm25_data = {
        "bm25": bm25, 
        "qids": qids, 
        "qid2idx": qid2idx,
        "num_docs": len(qids)
    }
    
    pickle.dump(bm25_data, open(args.out_bm25, "wb"))
    print(f"[build_bm25_filtered] Saved BM25 index to {args.out_bm25}")
    print(f"[build_bm25_filtered] Saved corpus to {args.out_corpus}")
    print(f"[build_bm25_filtered] Final: {len(qids):,} entities indexed")

if __name__ == "__main__":
    main()
