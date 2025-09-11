#!/usr/bin/env python3
import json, argparse, re, gzip
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from rank_bm25 import BM25Okapi

def gold_qid(s):
    return s.get("motif",[None]*5)[-1] if s.get("source")=="kg-qa-generated" else None

def toks(s):
    return set(re.sub(r"[^a-z0-9 ]+"," "," ".join(s.lower().split())).split())

def ngrams(s, n=3):
    return set(s[i:i+n] for i in range(len(s)-n+1))

def load_labels(path="data/raw/kg/wikidata5m/wikidata5m_entity.txt"):
    d={}
    with open(path,encoding="utf-8") as f:
        for ln in f:
            p=ln.rstrip("\n").split("\t")
            if len(p)>=2: d[p[0]]=p[1]
    return d

def load_descriptions(path="data/raw/kg/wikidata5m/wikidata5m_text.txt.gz"):
    d = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for ln in f:
            parts = ln.strip().split('\t')
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--entity_labels", default="data/raw/kg/wikidata5m/wikidata5m_entity.txt")
    ap.add_argument("--entity_descriptions", default="data/raw/kg/wikidata5m/wikidata5m_text.txt.gz")
    ap.add_argument("--train_frac", type=float, default=0.7, help="Fraction of samples for training")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for train/test split")
    args=ap.parse_args()
    qid2label=load_labels(args.entity_labels)
    qid2desc=load_descriptions(args.entity_descriptions)

    samples = [json.loads(ln) for ln in open(args.in_jsonl, encoding="utf-8")]
    
    # Split samples with gold QIDs into train/test
    np.random.seed(args.seed)
    gold_samples = [s for s in samples if gold_qid(s)]
    n_train = int(len(gold_samples) * args.train_frac)
    indices = np.random.permutation(len(gold_samples))
    train_indices = set(indices[:n_train])
    test_indices = set(indices[n_train:])
    
    train_samples = [gold_samples[i] for i in range(len(gold_samples)) if i in train_indices]
    test_samples = [gold_samples[i] for i in range(len(gold_samples)) if i in test_indices]
    
    print(f"Split: {len(train_samples)} train, {len(test_samples)} test samples")

    # 1. Prepare training data (only from train_samples)
    X_train, y_train = [], []
    for s in train_samples:
        g = gold_qid(s)

        q = (s.get("text") or s.get("question",""))
        qT = toks(q)
        
        T=s.get("text_candidates") or []
        K=s.get("kg_candidates") or []
        
        def norm(c):
            if not c: return {}
            m = max(x["score"] for x in c) or 1.0
            return {x["qid"]: x["score"]/m for x in c}
        
        tmap, kmap = norm(T), norm(K)
        all_qids = set(tmap) | set(kmap)
        if not all_qids: continue

        # Build description corpus for on-the-fly BM25
        candidate_qids = list(all_qids)
        desc_corpus = [(qid2label.get(qid, "") + " " + qid2desc.get(qid, "")).strip() for qid in candidate_qids]
        tokenized_desc_corpus = [doc.split() for doc in desc_corpus]
        bm25_desc = BM25Okapi(tokenized_desc_corpus)
        tokenized_query = q.split()
        desc_scores = bm25_desc.get_scores(tokenized_query)
        qid_to_desc_score = {qid: score for qid, score in zip(candidate_qids, desc_scores)}

        kg_hop_map = {c['qid']: c.get('hop', 10**9) for c in K}
        q_ngrams = ngrams(q.lower())

        for qid in all_qids:
            lab = qid2label.get(qid,"").lower()
            labT = toks(lab)
            lex_score = len(qT & labT) / max(1, len(labT))

            # Rich features
            desc_bm25_score = qid_to_desc_score.get(qid, 0.0)
            hop = kg_hop_map.get(qid, 10**9)
            hop_inv = 1.0 / (1.0 + hop)
            features = [
                tmap.get(qid, 0),
                kmap.get(qid, 0),
                lex_score,
                desc_bm25_score,
                hop_inv
            ]
            X_train.append(features)
            y_train.append(1 if qid == g else 0)

    # 2. Train Logistic Regression model
    print(f"Training Logistic Regression on {len(X_train)} candidates...")
    model = LogisticRegression(class_weight='balanced', solver='liblinear')
    model.fit(X_train, y_train)
    print("Model trained. Coefficients:", model.coef_)

    # 3. Evaluate on test set
    def evaluate_recall(samples_to_eval, model, qid2label, qid2desc):
        hits = {1: 0, 5: 0, 10: 0, 20: 0}
        total = 0
        
        for s in samples_to_eval:
            g = gold_qid(s)
            if not g: continue
            total += 1
            
            q = (s.get("text") or s.get("question",""))
            qT = toks(q)
            T = s.get("text_candidates") or []
            K = s.get("kg_candidates") or []
            
            def norm(c):
                if not c: return {}
                m = max(x["score"] for x in c) or 1.0
                return {x["qid"]: x["score"]/m for x in c}
            
            tmap, kmap = norm(T), norm(K)
            all_qids = set(tmap) | set(kmap)
            
            if all_qids:
                # Build description corpus for on-the-fly BM25
                candidate_qids = list(all_qids)
                desc_corpus = [(qid2label.get(qid, "") + " " + qid2desc.get(qid, "")).strip() for qid in candidate_qids]
                tokenized_desc_corpus = [doc.split() for doc in desc_corpus]
                bm25_desc = BM25Okapi(tokenized_desc_corpus)
                tokenized_query = q.split()
                desc_scores = bm25_desc.get_scores(tokenized_query)
                qid_to_desc_score = {qid: score for qid, score in zip(candidate_qids, desc_scores)}

                kg_hop_map = {c['qid']: c.get('hop', 10**9) for c in K}
                q_ngrams = ngrams(q.lower())

                features = []
                for qid in all_qids:
                    lab = qid2label.get(qid,"").lower()
                    labT = toks(lab)
                    lex_score = len(qT & labT) / max(1, len(labT))

                    # Rich features
                    desc_bm25_score = qid_to_desc_score.get(qid, 0.0)
                    hop = kg_hop_map.get(qid, 10**9)
                    hop_inv = 1.0 / (1.0 + hop)
                    features.append([
                        tmap.get(qid, 0),
                        kmap.get(qid, 0),
                        lex_score,
                        desc_bm25_score,
                        hop_inv
                    ])
                
                probabilities = model.predict_proba(features)[:, 1]
                ranked_qids = [qid for qid, _ in sorted(zip(all_qids, probabilities), key=lambda x: x[1], reverse=True)]
                
                for k in [1, 5, 10, 20]:
                    if g in ranked_qids[:k]:
                        hits[k] += 1
        
        return {k: hits[k] / max(1, total) for k in hits}

    test_recall = evaluate_recall(test_samples, model, qid2label, qid2desc)
    print(f"Test Recall: @1={test_recall[1]:.3f}, @5={test_recall[5]:.3f}, @10={test_recall[10]:.3f}, @20={test_recall[20]:.3f}")

    # 4. Rerank and write output
    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for s in samples:
            T=s.get("text_candidates") or []
            K=s.get("kg_candidates") or []
            
            def norm(c):
                if not c: return {}
                m = max(x["score"] for x in c) or 1.0
                return {x["qid"]: x["score"]/m for x in c}
            tmap, kmap = norm(T), norm(K)
            all_qids = set(tmap) | set(kmap)

            if all_qids:
                # Build description corpus for on-the-fly BM25
                candidate_qids = list(all_qids)
                desc_corpus = [(qid2label.get(qid, "") + " " + qid2desc.get(qid, "")).strip() for qid in candidate_qids]
                tokenized_desc_corpus = [doc.split() for doc in desc_corpus]
                bm25_desc = BM25Okapi(tokenized_desc_corpus)
                tokenized_query = q.split()
                desc_scores = bm25_desc.get_scores(tokenized_query)
                qid_to_desc_score = {qid: score for qid, score in zip(candidate_qids, desc_scores)}

                kg_hop_map = {c['qid']: c.get('hop', 10**9) for c in K}
                q_ngrams = ngrams(q.lower())

                rerank_features = []
                q = (s.get("text") or s.get("question",""))
                qT = toks(q)
                for qid in all_qids:
                    lab = qid2label.get(qid,"").lower()
                    labT = toks(lab)
                    lex_score = len(qT & labT) / max(1, len(labT))

                    # Rich features
                    desc_bm25_score = qid_to_desc_score.get(qid, 0.0)
                    hop = kg_hop_map.get(qid, 10**9)
                    hop_inv = 1.0 / (1.0 + hop)
                    rerank_features.append([
                        tmap.get(qid, 0),
                        kmap.get(qid, 0),
                        lex_score,
                        desc_bm25_score,
                        hop_inv
                    ])
                
                # Predict probability of being the correct answer
                probabilities = model.predict_proba(rerank_features)[:, 1]
                
                fused_scores = {qid: prob for qid, prob in zip(all_qids, probabilities)}
                s["fused_candidates"] = sorted([{"qid": q, "score": s} for q, s in fused_scores.items()], key=lambda x: x['score'], reverse=True)[:20]

            out.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote reranked output to {args.out_jsonl}")

if __name__=="__main__": main()
