#!/usr/bin/env python3
import csv, pickle, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

FEATS = ["n_mentions","n_linked_qids","n_conf_qids","avg_qid_score","nil_rate","two_hop_reachable"]
def load_xy(p):
    X, y = [], []
    with open(p, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            y.append(r["slice"])
            X.append([float(r[k]) for k in FEATS])
    return np.array(X), np.array(y)

def main():
    feats_csv = Path("data/phase2/router_dev_v5.csv")
    out = Path("data/phase2/router_v1.pkl")
    X, y = load_xy(feats_csv)
    keep = np.isin(y, ["kg","bridge","dense"])
    X, y = X[keep], y[keep]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=200, class_weight="balanced").fit(Xtr, ytr)
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"model": clf, "features": FEATS}, open(out, "wb"))
    print(f"Saved router → {out}")
if __name__ == "__main__":
    main()
