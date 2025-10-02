#!/usr/bin/env python3
import os, json, argparse, numpy as np, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/phase3/kge/transE_v1")
    ap.add_argument("--train_tsv", default="data/phase3/kge/train.tsv")
    ap.add_argument("--samples", type=int, default=200000)
    args = ap.parse_args()

    E = np.load(os.path.join(args.bundle, "e_emb.npy"))
    R = np.load(os.path.join(args.bundle, "r_emb.npy"))
    e_map = json.load(open(os.path.join(args.bundle, "e_map.json")))
    r_map = json.load(open(os.path.join(args.bundle, "r_map.json")))

    rows = open(args.train_tsv).read().splitlines()
    S = rows if len(rows) <= args.samples else random.sample(rows, args.samples)

    global_vals = []
    per_rel = {}  # r -> list of scores
    miss = 0
    for ln in S:
        h,r,t = ln.split("\t")
        r = r.split("/")[-1]
        ih, ir, it = e_map.get(h), r_map.get(r), e_map.get(t)
        if None in (ih, ir, it): 
            miss += 1; continue
        s = -np.linalg.norm(E[ih] + R[ir] - E[it], ord=1)
        global_vals.append(s)
        per_rel.setdefault(r, []).append(s)

    mu = float(np.mean(global_vals)); sd = float(np.std(global_vals) + 1e-6)
    stats = {"global": {"mu": mu, "sd": sd}, "per_relation": {}}
    for r, vals in per_rel.items():
        if len(vals) >= 50:  # require a bit of support
            stats["per_relation"][r] = {
                "mu": float(np.mean(vals)),
                "sd": float(np.std(vals) + 1e-6)
            }

    out = os.path.join(args.bundle, "kge_stats.json")
    json.dump(stats, open(out, "w"))
    print("Saved", out, "missed triples:", miss, "global:", stats["global"], 
          "per_rel_count:", len(stats["per_relation"]))

if __name__ == "__main__":
    main()
