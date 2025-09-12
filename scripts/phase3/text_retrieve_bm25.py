#!/usr/bin/env python3
import json, pickle, argparse, re
import numpy as np
from pathlib import Path
from tqdm import tqdm

def norm(text):
    text = re.sub(r"[^a-z0-9 ]+"," ", text.lower())
    return re.sub(r"\s+"," ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25_pkls", default="data/phase2/bm25_U.pkl,data/phase2/bm25_delta.pkl")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="optional cap on #queries for quicker runs")
    ap.add_argument("--force_process_all", action="store_true", help="Process all items regardless of route_v1")
    args = ap.parse_args()

    print(f"[INFO] Loading BM25 indices from {args.bm25_pkls}...")
    packs = []
    for pth in args.bm25_pkls.split(","):
        print(f"  - Loading {pth}")
        P = pickle.load(open(pth, "rb"))
        packs.append({"bm25": P["bm25"], "qids": P["qids"]})
    print(f"[INFO] {len(packs)} BM25 indices loaded.")

    def pack_topk(P, toks, k):
        scores = P["bm25"].get_scores(toks)
        if k >= len(scores):
            idx = np.argsort(scores)[::-1]
        else:
            idx = np.argpartition(scores, -k)[-k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
        return [(P["qids"][j], float(scores[j])) for j in idx]

    with open(args.in_jsonl, 'r', encoding='utf-8-sig') as f_in, open(args.out_jsonl, "w", encoding="utf-8") as out:
        for i, ln in enumerate(tqdm(f_in, desc="BM25 Retrieval")):
            if args.limit and i >= args.limit: break
            if not ln.strip(): continue
            try:
                s = json.loads(ln)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON on line {i+1}")
                continue

            if args.force_process_all or s.get("route_v1") in ("Dense","Both"):
                q = s.get("text") or s.get("question","")
                toks = norm(q).split()

                cand = {}
                for P in packs:
                    for qid, sc in pack_topk(P, toks, args.topk):
                        if sc > cand.get(qid, -1e9):
                            cand[qid] = sc
                
                top = sorted(cand.items(), key=lambda x: x[1], reverse=True)[:args.topk]
                s["text_candidates"] = [{"qid": qid, "score": float(sc)} for qid, sc in top]
            out.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n[INFO] Text retrieval complete. Output written to {args.out_jsonl}")
if __name__ == "__main__":
    main()
