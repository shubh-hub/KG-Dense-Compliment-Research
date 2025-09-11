#!/usr/bin/env python3
import json, pickle, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--features_csv", default="data/phase2/router_dev_v5.csv")
    ap.add_argument("--router_pkl", default="data/phase2/router_v1.pkl")
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    clf_pack = pickle.load(open(args.router_pkl,"rb"))
    clf, feats = clf_pack["model"], clf_pack["features"]
    fout = open(args.out_jsonl,"w",encoding="utf-8")

    with open(args.in_jsonl, encoding="utf-8") as f:
        for ln in f:
            x = json.loads(ln)
            r = x.get("router_feats", {})
            X = [[float(r.get(k,0.0)) for k in feats]]
            pred = clf.predict(X)[0]   # 'kg' | 'bridge' | 'dense'
            route = {"kg":"KG","bridge":"Both","dense":"Dense"}[pred]
            x["route_v1"] = route
            fout.write(json.dumps(x, ensure_ascii=False) + "\n")
    fout.close()
    print(f"Routed → {args.out_jsonl}")
if __name__ == "__main__":
    main()
