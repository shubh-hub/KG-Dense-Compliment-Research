#!/usr/bin/env python3
import json, argparse, math

def gold_qid(s):
    return s.get("motif",[None]*5)[-1] if s.get("source")=="kg-qa-generated" else None

def topk_qids(cands, k):
    if not cands: return []
    return [c["qid"] for c in sorted(cands, key=lambda x: x["score"], reverse=True)[:k]]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--ks", default="1,5,10,20")
    args=ap.parse_args()
    ks=[int(x) for x in args.ks.split(",")]

    N=0
    stats={"text":{k:0 for k in ks},"kg":{k:0 for k in ks},"fused":{k:0 for k in ks}}
    for ln in open(args.in_jsonl, encoding="utf-8"):
        s=json.loads(ln)
        if s.get("source")!="kg-qa-generated": continue
        g=gold_qid(s)
        if not g: continue
        N+=1
        T=s.get("text_candidates") or []
        K=s.get("kg_candidates") or []
        F = s.get("fused_candidates") or []
        # if fused_candidates is not present, do the simple fusion for backward compatibility
        if not F:
            def norm(c):
                if not c: return {}
                m=max(x["score"] for x in c) or 1.0
                return {x["qid"]: x["score"]/m for x in c}
            tmap, kmap = norm(T), norm(K)
            union = {q: tmap.get(q,0.0)+kmap.get(q,0.0) for q in set(tmap)|set(kmap)}
            F = [{"qid":q,"score":sc} for q,sc in union.items()]

        for k in ks:
            if g in topk_qids(T,k): stats["text"][k]+=1
            if g in topk_qids(K,k): stats["kg"][k]+=1
            if g in topk_qids(F,k): stats["fused"][k]+=1

    def pct(a): return f"{(100.0*a/max(1,N)):.1f}%"
    print(f"N={N}")
    for ch in ["text","kg","fused"]:
        line=", ".join([f"@{k}={pct(stats[ch][k])}" for k in ks])
        print(f"{ch}: {line}")

if __name__=="__main__": main()
