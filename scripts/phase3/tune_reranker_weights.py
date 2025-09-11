#!/usr/bin/env python3
import json, argparse, re, numpy as np
from collections import defaultdict

def gold_qid(s):
    return s.get("motif",[None]*5)[-1] if s.get("source")=="kg-qa-generated" else None

def toks(s):
    return set(re.sub(r"[^a-z0-9 ]+"," "," ".join(s.lower().split())).split())

def load_labels(path="data/raw/kg/wikidata5m/wikidata5m_entity.txt"):
    d={}
    with open(path,encoding="utf-8") as f:
        for ln in f:
            p=ln.rstrip("\n").split("\t")
            if len(p)>=2: d[p[0]]=p[1]
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--entity_labels", default="data/raw/kg/wikidata5m/wikidata5m_entity.txt")
    args=ap.parse_args()
    qid2label=load_labels(args.entity_labels)

    samples = [json.loads(ln) for ln in open(args.in_jsonl, encoding="utf-8")]
    
    # Pre-calculate scores
    for s in samples:
        q = (s.get("text") or s.get("question",""))
        qT = toks(q)
        
        T=s.get("text_candidates") or []
        K=s.get("kg_candidates") or []
        
        def norm(c):
            if not c: return {}
            m = max(x["score"] for x in c) or 1.0
            return {x["qid"]: x["score"]/m for x in c}
        
        s["_tmap"] = norm(T)
        s["_kmap"] = norm(K)
        
        s["_lexmap"] = {}
        all_qids = set(s["_tmap"]) | set(s["_kmap"])
        for qid in all_qids:
            lab = qid2label.get(qid,"").lower()
            labT = toks(lab)
            s["_lexmap"][qid] = len(qT & labT) / max(1, len(labT))

    best_recall = -1
    best_weights = None

    # Grid search over weights
    for w_text in np.arange(0.0, 1.01, 0.1):
        for w_kg in np.arange(0.0, 1.01 - w_text, 0.1):
            w_lex = 1.0 - w_text - w_kg
            weights = (w_text, w_kg, w_lex)
            
            hits_at_10 = 0
            N = 0

            for s in samples:
                g = gold_qid(s)
                if not g: continue
                N += 1
                
                union = {}
                all_qids = set(s["_tmap"]) | set(s["_kmap"])
                for qid in all_qids:
                    score = (weights[0] * s["_tmap"].get(qid, 0) + 
                             weights[1] * s["_kmap"].get(qid, 0) + 
                             weights[2] * s["_lexmap"].get(qid, 0))
                    union[qid] = score
                
                if not union: continue
                
                top_10_qids = [q for q, s in sorted(union.items(), key=lambda item: item[1], reverse=True)[:10]]
                if g in top_10_qids:
                    hits_at_10 += 1

            recall_at_10 = hits_at_10 / max(1, N)
            print(f"Weights (T/K/L): ({w_text:.1f}/{w_kg:.1f}/{w_lex:.1f}) -> Recall@10: {recall_at_10:.3f}")

            if recall_at_10 > best_recall:
                best_recall = recall_at_10
                best_weights = weights

    print("\n-----------------------------------")
    print(f"Best Recall@10: {best_recall:.3f}")
    print(f"Best Weights (Text/KG/Lex): ({best_weights[0]:.1f}/{best_weights[1]:.1f}/{best_weights[2]:.1f})")
    print("-----------------------------------")

if __name__=="__main__": main()
