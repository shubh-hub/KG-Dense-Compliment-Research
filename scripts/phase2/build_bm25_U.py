#!/usr/bin/env python3
import argparse, json, gzip, re, pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

def open_any(p):
    return gzip.open(p, "rt", encoding="utf-8", errors="ignore") if str(p).endswith(".gz") else open(p, "r", encoding="utf-8", errors="ignore")

def norm(text):
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+"," ", text.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--U_qids", required=True)
    ap.add_argument("--entity_labels", required=True, help="wikidata5m_entity.txt")
    ap.add_argument("--entity_desc", required=True, help="wikidata5m_text.txt.gz")
    ap.add_argument("--out_corpus", required=True)
    ap.add_argument("--out_bm25", required=True)
    args = ap.parse_args()

    U = {ln.strip() for ln in open(args.U_qids) if ln.strip()}
    print(f"[bm25_U] U size: {len(U):,}")

    # qid -> label (only U)
    qid2label = {}
    with open(args.entity_labels, encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 2: continue
            qid, label = parts[0], parts[1]
            if qid in U: qid2label[qid] = label
            if i % 1_000_000 == 0:
                print(f"[bm25_U] labels scanned: {i:,} | kept: {len(qid2label):,}")

    docs_tokens, qids = [], []
    Path(args.out_corpus).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_corpus, "w", encoding="utf-8") as out, open_any(args.entity_desc) as fin:
        kept = 0
        for j, ln in enumerate(fin, 1):
            parts = ln.rstrip("\n").split("\t", 1)
            if len(parts) != 2: continue
            qid, desc = parts
            if qid not in U: continue
            label = qid2label.get(qid, "")
            text = (label + " " + desc).strip() if label else desc
            out.write(json.dumps({"qid": qid, "text": text}, ensure_ascii=False) + "\n")
            qids.append(qid)
            docs_tokens.append(norm(text).split())
            kept += 1
            if j % 1_000_000 == 0:
                print(f"[bm25_U] desc scanned: {j:,} | kept: {kept:,}")

    print(f"[bm25_U] Building BM25 for {len(qids):,} docs")
    bm25 = BM25Okapi(docs_tokens)
    qid2idx = {q:i for i,q in enumerate(qids)}
    with open(args.out_bm25, "wb") as f:
        pickle.dump({"bm25": bm25, "qids": qids, "qid2idx": qid2idx, "num_docs": len(qids)}, f)
    print(f"[bm25_U] Saved: {args.out_bm25}  | corpus: {args.out_corpus}")
if __name__ == "__main__":
    main()
