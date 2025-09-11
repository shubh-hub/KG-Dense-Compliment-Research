#!/usr/bin/env python3
import json, argparse
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)  # use the routed or original combined file; must contain the 1k kg-qa
    ap.add_argument("--u_qids", required=True)    # existing: data/phase2/U_qids.txt
    ap.add_argument("--out_u_qids", required=True)
    args=ap.parse_args()

    U=set(ln.strip() for ln in open(args.u_qids) if ln.strip())
    added=set()
    with open(args.in_jsonl, encoding="utf-8") as f:
        for ln in f:
            s=json.loads(ln)
            if s.get("source")!="kg-qa-generated": continue
            m=s.get("motif",[])
            if len(m)>=5:
                h,r1,m1,r2,t = m
                for q in (h,m1,t):
                    if q not in U:
                        U.add(q); added.add(q)
    Path(args.out_u_qids).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_u_qids,"w") as fo:
        for q in sorted(U): fo.write(q+"\n")
    print(f"Expanded U: +{len(added)} kg-qa QIDs -> total {len(U)}")

if __name__=="__main__":
    main()
