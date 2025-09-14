#!/usr/bin/env python3
import json, argparse, re, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from tqdm import tqdm

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

def get_features(s, qid, qid2label, qT, tmap, kmap, k_cand_map):
    """Extracts a dictionary of features for a given sample and candidate QID."""
    features = {}
    # Candidate-level features
    features['text_score_norm'] = tmap.get(qid, 0)
    features['kg_score_norm'] = kmap.get(qid, 0)
    
    lab = qid2label.get(qid,"").lower()
    labT = toks(lab)
    features['lex_score'] = len(qT & labT) / max(1, len(labT))
    
    features['in_text'] = 1 if qid in tmap else 0
    features['in_kg'] = 1 if qid in kmap else 0
    if qid in k_cand_map:
        features['kg_hop'] = k_cand_map[qid]['hop']

    # Sample-level features (prefixed with s_)
    meta = s.get('allowlist_meta', {})
    features['s_allowlist_enabled'] = 1 if meta.get('enabled') else 0
    features['s_allowlist_size'] = meta.get('size', 0)
    features['s_allowlist_pre_cap_size'] = meta.get('pre_cap_size', 0)
    features['s_q_allow_present'] = 1 if meta.get('q_allow_size', 0) > 0 else 0
    features['s_s_allow_present'] = 1 if meta.get('s_allow_size', 0) > 0 else 0
    features['s_both_allow_present'] = 1 if features['s_q_allow_present'] and features['s_s_allow_present'] else 0

    kg_cands = s.get('kg_candidates', [])
    if kg_cands:
        scores = [c['score'] for c in kg_cands]
        features['s_kg_max_score'] = np.max(scores)
        features['s_kg_mean_score'] = np.mean(scores)
        hops = [c['hop'] for c in kg_cands]
        features['s_kg_min_hop'] = np.min(hops)
        for i in range(1, 5):
            features[f's_kg_hop_{i}_count'] = hops.count(i)

    return features

def apply_monotone_guard(kg_qids_topk, ranked_qids, k=20):
    """
    Ensure recall monotonicity: all items in kg_qids_topk[:k] appear in ranked_qids[:k].
    ranked_qids is a list sorted by fused score (desc).
    """
    S = set(kg_qids_topk[:k])
    top = ranked_qids[:k]
    missing = [q for q in kg_qids_topk[:k] if q not in set(top)]
    if missing:
        # Evict from the tail first: worst-ranked non-KG items
        evict_positions = [i for i in range(k-1, -1, -1) if top[i] not in S]
        for q in missing:
            if not evict_positions:
                break
            i = evict_positions.pop(0)  # replace lowest-ranked non-KG
            top[i] = q
    return top

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--entity_labels", default="data/raw/kg/wikidata5m/wikidata5m_entity.txt")
    ap.add_argument("--train_frac", type=float, default=0.8, help="Fraction of samples for training")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for train/test split")
    ap.add_argument("--save_model_pkl", help="Path to save the trained model")
    ap.add_argument("--load_model_pkl", help="Path to load a pre-trained model")
    ap.add_argument("--monotone_guard", action="store_true", help="Keep all KG@k items inside fused@k after reranking.")
    ap.add_argument("--inspect_model", action="store_true", help="If set, load model and print feature importance.")
    args=ap.parse_args()
    qid2label=load_labels(args.entity_labels)

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
    feature_map = {}
    X_train, y_train = [], []
    print("Preparing training data...")
    for s in tqdm(train_samples):
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
        k_cand_map = {c['qid']: c for c in K}
        all_qids = set(tmap) | set(kmap)
        if not all_qids: continue

        for qid in all_qids:
            features = get_features(s, qid, qid2label, qT, tmap, kmap, k_cand_map)
            # Dynamically build feature map from first sample's features
            if not feature_map:
                feature_map = {name: i for i, name in enumerate(features.keys())}
            
            feature_vector = [0] * len(feature_map)
            for name, value in features.items():
                if name in feature_map:
                    feature_vector[feature_map[name]] = value
            
            X_train.append(feature_vector)
            y_train.append(1 if qid == g else 0)

    # 2. Train or load Logistic Regression model
    if args.load_model_pkl:
        print(f"Loading pre-trained model from {args.load_model_pkl}...")
        with open(args.load_model_pkl, 'rb') as f:
            bundle = pickle.load(f)
            model = bundle['model']
            feature_map = bundle['feature_map']
    else:
        if args.train_frac > 0:
            print(f"Training Logistic Regression on {len(X_train)} candidates...")
            model = LogisticRegression(class_weight='balanced', solver='liblinear')
            model.fit(X_train, y_train)
            print("Model trained. Coefficients:", model.coef_)
            if args.save_model_pkl:
                print(f"Saving model and feature map to {args.save_model_pkl}...")
                bundle = {'model': model, 'feature_map': feature_map}
                with open(args.save_model_pkl, 'wb') as f:
                    pickle.dump(bundle, f)
        else:
            model, feature_map = None, None # No training needed

    if args.inspect_model:
        if not model or not feature_map:
            print("[ERROR] Cannot inspect model. Please provide a trained model with --load_model_pkl.")
            return
        print("\n--- Feature Importance ---")
        feature_names = sorted(feature_map.keys(), key=lambda k: feature_map[k])
        for name, coef in sorted(zip(feature_names, model.coef_[0]), key=lambda x: -abs(x[1])):
            print(f"{name:<30} {coef:.4f}")
        return # End execution after inspection

    # 3. Evaluate on test set
    def evaluate_recall(samples_to_eval, model, qid2label, feature_map):
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
            k_cand_map = {c['qid']: c for c in K}
            all_qids = set(tmap) | set(kmap)
            
            if all_qids:
                feature_vectors = []
                for qid in all_qids:
                    features = get_features(s, qid, qid2label, qT, tmap, kmap, k_cand_map)
                    
                    feature_vector = [0] * len(feature_map)
                    for name, value in features.items():
                        if name in feature_map:
                            feature_vector[feature_map[name]] = value
                    
                    feature_vectors.append(feature_vector)
                
                if feature_vectors:
                    probabilities = model.predict_proba(feature_vectors)[:, 1]
                    ranked_qids = [qid for qid, _ in sorted(zip(all_qids, probabilities), key=lambda x: x[1], reverse=True)]
                else:
                    ranked_qids = []
                
                for k in [1, 5, 10, 20]:
                    if g in ranked_qids[:k]:
                        hits[k] += 1
        
        return {k: hits[k] / max(1, total) for k in hits}

    if model and feature_map:
        test_recall = evaluate_recall(test_samples, model, qid2label, feature_map)
        print(f"Test Recall: @1={test_recall[1]:.3f}, @5={test_recall[5]:.3f}, @10={test_recall[10]:.3f}, @20={test_recall[20]:.3f}")

    # 4. Rerank and write output
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for s in tqdm(samples, desc="Reranking all samples"):
            T=s.get("text_candidates") or []
            K=s.get("kg_candidates") or []
            
            def norm(c):
                if not c: return {}
                m = max(x["score"] for x in c) or 1.0
                return {x["qid"]: x["score"]/m for x in c}
            tmap, kmap = norm(T), norm(K)
            all_qids = set(tmap) | set(kmap)

            if all_qids:
                rerank_features = []
                q = (s.get("text") or s.get("question",""))
                qT = toks(q)
                k_cand_map = {c['qid']: c for c in K}
                for qid in all_qids:
                    features = get_features(s, qid, qid2label, qT, tmap, kmap, k_cand_map)
                    feature_vector = [0] * len(feature_map)
                    for name, value in features.items():
                        if name in feature_map:
                            feature_vector[feature_map[name]] = value
                    rerank_features.append(feature_vector)
                
                if rerank_features and model:
                    probabilities = model.predict_proba(rerank_features)[:, 1]
                    fused_scores = {qid: prob for qid, prob in zip(all_qids, probabilities)}
                    ranked_qids = [q for q, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]

                    if args.monotone_guard:
                        kg_qids = [c['qid'] for c in K]
                        ranked_qids_topk = apply_monotone_guard(kg_qids, ranked_qids, k=20)
                        s["fused_candidates"] = [{"qid": q, "score": fused_scores.get(q, 0.0)} for q in ranked_qids_topk]
                        # Sanity Check
                        kg_topk = {c['qid'] for c in K[:20]}
                        fused_topk = {c['qid'] for c in s["fused_candidates"][:20]}
                        assert kg_topk.issubset(fused_topk), "Monotone guard violated"
                    else:
                        s["fused_candidates"] = [{"qid": q, "score": fused_scores[q]} for q in ranked_qids[:20]]
                else:
                    s["fused_candidates"] = []

            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote reranked output to {args.out_jsonl}")

if __name__=="__main__": main()
