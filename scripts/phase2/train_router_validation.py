#!/usr/bin/env python3
import csv
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    features_path = Path("data/phase2/router_dev_v5.csv")
    if not features_path.exists():
        print(f"ERROR: Features file not found at {features_path}")
        return

    X = []
    y = []
    print(f"Loading features from {features_path}...")
    with open(features_path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            y.append(r["slice"])
            X.append([
                float(r["n_mentions"]),
                float(r["n_linked_qids"]),
                float(r["n_conf_qids"]),
                float(r["avg_qid_score"]),
                float(r["nil_rate"]),
                float(r["two_hop_reachable"])
            ])
    
    X = np.array(X)
    y = np.array(y)

    # Collapse to 3-way classification: kg / bridge / dense
    keep = np.isin(y, ["kg", "bridge", "dense"])
    X, y = X[keep], y[keep]
    print(f"Training router on {len(y)} samples.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=200, multi_class="auto", class_weight='balanced').fit(X_train, y_train)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, clf.predict(X_test), digits=3))

if __name__ == "__main__":
    main()
