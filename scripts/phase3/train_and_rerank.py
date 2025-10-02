#!/usr/bin/env python3
import json, argparse, re, pickle, os, math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import scipy.stats as ss
from collections import defaultdict
from tqdm import tqdm

def canon_id(x):
    if not x: return None
    x = x.rsplit('/', 1)[-1]
    return x.split(':', 1)[-1]

def load_kge_bundle(kge_dir):
    E = np.load(os.path.join(kge_dir, "e_emb.npy"))
    R = np.load(os.path.join(kge_dir, "r_emb.npy"))
    e_map = json.load(open(os.path.join(kge_dir, "e_map.json")))
    r_map = json.load(open(os.path.join(kge_dir, "r_map.json")))
    stats = json.load(open(os.path.join(kge_dir, "kge_stats.json")))
    return {
        "E": E, "R": R, "e_map": e_map, "r_map": r_map,
        "g_mu": stats["global"]["mu"], "g_sd": stats["global"]["sd"],
        "per_r": stats.get("per_relation", {})
    }

def kge_z(h, r, t, K):
    h, r, t = canon_id(h), canon_id(r), canon_id(t)
    if None in (h, r, t): return None
    ih, ir, it = K["e_map"].get(h), K["r_map"].get(r), K["e_map"].get(t)
    if None in (ih, ir, it): return None
    s = -np.linalg.norm(K["E"][ih] + K["R"][ir] - K["E"][it], ord=1)
    rs = K["per_r"].get(r)
    mu = rs["mu"] if rs else K["g_mu"]
    sd = rs["sd"] if rs else K["g_sd"]
    if sd <= 1e-8: return 0.0
    z = (s - mu) / sd
    return float(np.clip(z, -2.0, 2.0))

def get_gold_qid(s):
    g = (s.get("motif", [None]*5))[-1]
    if not g:
        if "answer_qids" in s and s["answer_qids"]:
            g = s["answer_qids"][0]
    return g

def toks(s):
    return set(re.sub(r"[^a-z0-9 ]+"," "," ".join(s.lower().split())).split())

def load_labels(path="data/raw/kg/wikidata5m/wikidata5m_entity.txt"):
    d={}
    with open(path,encoding="utf-8") as f:
        for ln in f:
            p=ln.rstrip("\n").split("\t")
            if len(p)>=2: d[p[0]]=p[1]
    return d

def is_gold(s):
    g = (s.get("motif", [None]*5))[-1]
    if not g:
        if "answer_qids" in s and s["answer_qids"]:
            g = s["answer_qids"][0]
        else:
            return False
    return g is not None

def get_features(s, qid, qid2label, qT, tmap, kmap, k_cand_map, KGE=None, kge_rel_cap=32):
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
        features['kg_score'] = k_cand_map[qid].get('score', -99.)
        features['kg_hop'] = k_cand_map[qid].get('hop', 0)

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
        hops = [c.get('hop', 0) for c in kg_cands]
        features['s_kg_min_hop'] = np.min(hops)
        for i in range(1, 5):
            features[f's_kg_hop_{i}_count'] = hops.count(i)

    # --- KGE features (post-walk) ---
    if KGE:
        seeds = s.get("seeds") or []
        allow_rels = s.get("allowlist_rels") or []
        if allow_rels and len(allow_rels) > kge_rel_cap:
            allow_rels = allow_rels[:kge_rel_cap]

        cand_info = k_cand_map.get(qid, {})
        anchor = cand_info.get("best_seed") or (seeds[0] if seeds else None)
        last_rel = cand_info.get("last_rel")
        
        features['kg_path_log_prior'] = cand_info.get('path_log_prior', 0.0)
        z_vals = []

        if anchor and last_rel:
            z_lr = kge_z(anchor, last_rel, qid, KGE)
            if z_lr is not None:
                features['kge_z_last'] = z_lr
                z_vals.append(z_lr)

        if anchor and allow_rels:
            best = max((kge_z(anchor, r, qid, KGE) for r in allow_rels if kge_z(anchor, r, qid, KGE) is not None), default=None)
            if best is not None:
                features['kge_z_max_allow'] = best
                z_vals.append(best)

        if seeds:
            per_seed = []
            for h in seeds:
                z_h = None
                if last_rel:
                    z_h = kge_z(h, last_rel, qid, KGE)
                if (z_h is None) and allow_rels:
                    z_h = max((kge_z(h, r, qid, KGE) for r in allow_rels if kge_z(h, r, qid, KGE) is not None), default=None)
                if z_h is not None:
                    per_seed.append(z_h)
            if per_seed:
                features['kge_z_seed_max']  = float(np.max(per_seed))
                features['kge_z_seed_mean'] = float(np.mean(per_seed))
                z_vals.extend(per_seed)

        if z_vals:
            features['kge_pos'] = 1 if max(z_vals) > 0 else 0
        
        for name in ['kge_z_last','kge_z_max_allow','kge_z_seed_max','kge_z_seed_mean','kge_pos']:
            features.setdefault(name, 0.0)

    return features

def add_within_sample_features(feats_per_qid, score_names_to_rank):
    if not feats_per_qid: return
    qids = list(feats_per_qid.keys())
    for feat_name in score_names_to_rank:
        try:
            vals = np.array([feats_per_qid[q].get(feat_name, 0.0) for q in qids])
            ranks = ss.rankdata(vals, method='average')
            for i, qid in enumerate(qids):
                feats_per_qid[qid][f"{feat_name}_rank"] = ranks[i]
                feats_per_qid[qid][f"{feat_name}_pct"] = ranks[i] / max(1, len(qids))
        except (KeyError, ValueError) as e:
            print(f"[WARN] Could not compute rank for {feat_name}: {e}")

def apply_monotone_guard(kg_qids_topk, ranked_qids, k=20, guard_k=10):
    """
    Ensure recall monotonicity: all items in kg_qids_topk[:k] appear in ranked_qids[:k].
    ranked_qids is a list sorted by fused score (desc).
    """
    S = set(kg_qids_topk[:guard_k])
    top = ranked_qids[:k]
    missing = [q for q in S if q not in set(top)]
    if missing:
        # Evict from the tail first: worst-ranked non-KG items
        evict_positions = [i for i in range(k-1, -1, -1) if top[i] not in S]
        for q in missing:
            if not evict_positions:
                break
            i = evict_positions.pop(0)  # replace lowest-ranked non-KG
            top[i] = q
    return top

def evaluate_recall(samples_to_eval, model, qid2label, feature_map, KGE, args):
    hits = {1: 0, 5: 0, 10: 0, 20: 0}
    total = 0
    
    for s in samples_to_eval:
        g = get_gold_qid(s)
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
            feats_per_qid = {qid: get_features(s, qid, qid2label, qT, tmap, kmap, k_cand_map, KGE, kge_rel_cap=args.kge_rel_cap) for qid in all_qids}
            add_within_sample_features(feats_per_qid, ['text_score_norm', 'kg_score_norm', 'kge_z_seed_max', 'kge_z_seed_mean'])

            sorted_qids = sorted(list(all_qids))
            feature_vectors = []
            for qid in sorted_qids:
                features = feats_per_qid[qid]
                feature_vector = [0] * len(feature_map)
                for name, value in features.items():
                    if name in feature_map:
                        feature_vector[feature_map[name]] = value
                feature_vectors.append(feature_vector)
            
            if feature_vectors:
                probabilities = model.predict_proba(feature_vectors)[:, 1]
                ranked_qids = [qid for qid, _ in sorted(zip(sorted_qids, probabilities), key=lambda x: x[1], reverse=True)]
            else:
                ranked_qids = []
            
            for k in [1, 5, 10, 20]:
                if g in ranked_qids[:k]:
                    hits[k] += 1
    
    return {k: hits[k] / max(1, total) for k in hits}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--entity_labels", default="data/raw/kg/wikidata5m/wikidata5m_entity.txt")
    ap.add_argument("--n_splits", type=int, default=5, help="Number of splits for cross-validation")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for train/test split")
    ap.add_argument("--save_model_pkl", help="Path to save the trained model")
    ap.add_argument("--load_model_pkl", help="Path to load a pre-trained model")
    ap.add_argument("--monotone_guard", action="store_true", help="Keep all KG@k items inside fused@k after reranking.")
    ap.add_argument("--guard_k", type=int, default=10, help="Number of top KG candidates to protect with monotone guard.")
    ap.add_argument("--inspect_model", action="store_true", help="If set, load model and print feature importance.")
    ap.add_argument("--kge_dir", default=None, help="Dir with {e_emb.npy,r_emb.npy,e_map.json,r_map.json,kge_stats.json}")
    ap.add_argument("--kge_rel_cap", type=int, default=32, help="Cap on allowlist relations to try if last_rel missing")
    args=ap.parse_args()
    qid2label=load_labels(args.entity_labels)

    samples = [json.loads(ln) for ln in open(args.in_jsonl, encoding="utf-8")]
    KGE = load_kge_bundle(args.kge_dir) if args.kge_dir else None

    # If loading a model, we will use it for reranking all samples. Otherwise, we train.
    if args.load_model_pkl:
        print(f"\n--- Loading Pre-trained Model for Reranking ---")
        print(f"Loading model from {args.load_model_pkl}...")
        with open(args.load_model_pkl, 'rb') as f:
            bundle = pickle.load(f)
            model = bundle['model']
            feature_map = bundle['feature_map']
    else:
        # Perform k-fold cross-validation on gold samples to get a performance estimate
        gold_samples = [s for s in samples if is_gold(s)]
        if not gold_samples:
            print("[ERROR] No gold samples found for training or evaluation.")
            model, feature_map = None, None
        else:
            print(f"Performing {args.n_splits}-fold cross-validation on {len(gold_samples)} gold samples...")
            kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
            fold_recalls = []

            for fold, (train_indices, test_indices) in enumerate(kf.split(gold_samples)):
                print(f"\n--- Fold {fold+1}/{args.n_splits} ---")
                train_samples = [gold_samples[i] for i in train_indices]
                test_samples = [gold_samples[i] for i in test_indices]
                print(f"Split: {len(train_samples)} train, {len(test_samples)} test samples")

                # 1. Prepare training data for this fold
                all_feature_names = set()
                for s in tqdm(train_samples, desc="Pass 1: Collect features"):
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
                        features = get_features(s, qid, qid2label, qT, tmap, kmap, k_cand_map, KGE, kge_rel_cap=args.kge_rel_cap)
                        all_feature_names.update(features.keys())
                feature_map = {name: i for i, name in enumerate(sorted(list(all_feature_names)))}

                X_train, y_train = [], []
                for s in tqdm(train_samples, desc="Pass 2: Vectorize"):
                    g = get_gold_qid(s)
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
                    feats_per_qid = {qid: get_features(s, qid, qid2label, qT, tmap, kmap, k_cand_map, KGE, kge_rel_cap=args.kge_rel_cap) for qid in all_qids}
                    add_within_sample_features(feats_per_qid, ['text_score_norm', 'kg_score_norm', 'kge_z_seed_max', 'kge_z_seed_mean'])
                    
                    sorted_qids_train = sorted(list(all_qids))
                    for qid in sorted_qids_train:
                        features = feats_per_qid[qid]
                        feature_vector = [0] * len(feature_map)
                        for name, value in features.items():
                            if name in feature_map:
                                feature_vector[feature_map[name]] = value
                        X_train.append(feature_vector)
                        y_train.append(1 if qid == g else 0)

                # 2. Train Logistic Regression model for this fold
                model = make_pipeline(
                    StandardScaler(with_mean=True, with_std=True),
                    LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=200)
                )
                model.fit(X_train, y_train)

                # 3. Evaluate on test set for this fold
                test_recall = evaluate_recall(test_samples, model, qid2label, feature_map, KGE, args)
                fold_recalls.append(test_recall)
                print(f"Fold {fold+1} Recall: @1={test_recall[1]:.3f}, @5={test_recall[5]:.3f}, @10={test_recall[10]:.3f}, @20={test_recall[20]:.3f}")

            # --- End of cross-validation loop ---
            print("\n--- Cross-Validation Summary ---")
            avg_recall = defaultdict(float)
            for recall_dict in fold_recalls:
                for k, v in recall_dict.items():
                    avg_recall[k] += v
            for k in sorted(avg_recall.keys()):
                avg_recall[k] /= len(fold_recalls)
            print(f"Average Recall: @1={avg_recall[1]:.3f}, @5={avg_recall[5]:.3f}, @10={avg_recall[10]:.3f}, @20={avg_recall[20]:.3f}")

        # Train a final model on ALL gold data, using a feature map from the ENTIRE dataset
        print("\n--- Training Final Model on All Gold Data ---")
        print("Building global feature map from all samples...")
        all_feature_names = set()
        for s in tqdm(samples, desc="Global Features Pass"): # Iterate over ALL samples
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
            feats_per_qid = {qid: get_features(s, qid, qid2label, qT, tmap, kmap, k_cand_map, KGE, kge_rel_cap=args.kge_rel_cap) for qid in all_qids}
            add_within_sample_features(feats_per_qid, ['text_score_norm', 'kg_score_norm', 'kge_z_seed_max', 'kge_z_seed_mean'])
            for qid in all_qids:
                all_feature_names.update(feats_per_qid[qid].keys())
        
        feature_map = {name: i for i, name in enumerate(sorted(list(all_feature_names)))}
        print(f"Global feature map built with {len(feature_map)} features.")

        X_train, y_train = [], []
        print("Vectorizing training data using global feature map...")
        for s in tqdm(gold_samples, desc="Vectorize Gold Samples"):
            g = get_gold_qid(s)
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
            feats_per_qid = {qid: get_features(s, qid, qid2label, qT, tmap, kmap, k_cand_map, KGE, kge_rel_cap=args.kge_rel_cap) for qid in all_qids}
            add_within_sample_features(feats_per_qid, ['text_score_norm', 'kg_score_norm', 'kge_z_seed_max', 'kge_z_seed_mean'])
            
            sorted_qids_final_train = sorted(list(all_qids))
            for qid in sorted_qids_final_train:
                features = feats_per_qid[qid]
                feature_vector = [0.0] * len(feature_map)
                for name, value in features.items():
                    if name in feature_map:
                        feature_vector[feature_map[name]] = value
                X_train.append(feature_vector)
                y_train.append(1 if qid == g else 0)

        print(f"Training final model on {len(X_train)} candidates...")
        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=200)
        )
        model.fit(X_train, y_train)
        print("Final model trained.")

        if args.save_model_pkl:
            print(f"Saving final model and feature map to {args.save_model_pkl}...")
            bundle = {'model': model, 'feature_map': feature_map}
            with open(args.save_model_pkl, 'wb') as f:
                pickle.dump(bundle, f)

    if args.inspect_model:
        if not model or not feature_map:
            print("[ERROR] Cannot inspect model. No model was trained or loaded.")
            return
        print("\n--- Feature Importance (Final Model) ---")
        feature_names = sorted(feature_map.keys(), key=lambda k: feature_map[k])
        lr_model = model.named_steps['logisticregression']
        for name, coef in sorted(zip(feature_names, lr_model.coef_[0]), key=lambda x: -abs(x[1])):
            print(f"{name:<30} {coef:.4f}")
        return # End execution after inspection


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
                q = (s.get("text") or s.get("question",""))
                qT = toks(q)
                k_cand_map = {c['qid']: c for c in K}

                feats_per_qid = {qid: get_features(s, qid, qid2label, qT, tmap, kmap, k_cand_map, KGE, kge_rel_cap=args.kge_rel_cap) for qid in all_qids}
                add_within_sample_features(feats_per_qid, ['text_score_norm', 'kg_score_norm', 'kge_z_seed_max', 'kge_z_seed_mean'])

                # CRITICAL: Ensure a consistent order for features and scores
                sorted_qids = sorted(list(all_qids))
                rerank_features = []
                for qid in sorted_qids:
                    features = feats_per_qid[qid]
                    feature_vector = [0.0] * len(feature_map)
                    for name, value in features.items():
                        if name in feature_map:
                            idx = feature_map[name]
                            feature_vector[idx] = value
                    rerank_features.append(feature_vector)
                
                if rerank_features and model:
                    probabilities = model.predict_proba(rerank_features)[:, 1]
                    fused_scores = {qid: prob for qid, prob in zip(sorted_qids, probabilities)}
                    ranked_qids = [q for q, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]

                    if args.monotone_guard:
                        kg_qids = [c['qid'] for c in K]
                        ranked_qids_topk = apply_monotone_guard(kg_qids, ranked_qids, k=20, guard_k=args.guard_k)
                        s["fused_candidates"] = [{"qid": q, "score": fused_scores.get(q, 0.0)} for q in ranked_qids_topk]
                        # Sanity Check
                        kg_topk_set = {c['qid'] for c in K[:args.guard_k]}
                        fused_topk_set = {c['qid'] for c in s["fused_candidates"][:20]}
                        assert kg_topk_set.issubset(fused_topk_set), f"Monotone guard violated: {kg_topk_set - fused_topk_set}"

                    else:
                        s["fused_candidates"] = [{"qid": q, "score": fused_scores[q]} for q in ranked_qids[:20]]
                else:
                    s["fused_candidates"] = []

            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote reranked output to {args.out_jsonl}")

if __name__=="__main__": main()
