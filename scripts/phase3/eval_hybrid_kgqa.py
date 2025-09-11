#!/usr/bin/env python3
import json, argparse, math
from collections import defaultdict

def gold_qid(sample):
    if sample.get("source")=="kg-qa-generated":
        motif = sample.get("motif") or []
        if len(motif)>=5:
            return motif[-1]  # end entity
    return None

def top1(cands):
    if not cands: return None, -math.inf
    cands = sorted(cands, key=lambda x: x["score"], reverse=True)
    return cands[0]["qid"], cands[0]["score"]

def normalize(cands):
    if not cands: return {}
    mx = max(x["score"] for x in cands) or 1.0
    return {x["qid"]: (x["score"]/mx) for x in cands}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True) # output of text_retrieve step
    args = ap.parse_args()

    n=0; text_ok=0; kg_ok=0; fusion_ok=0; oracle_ok=0
    ci_count=0  # fusion correct & both wrong
    for ln in open(args.in_jsonl, encoding="utf-8"):
        s = json.loads(ln)
        g = gold_qid(s)
        if not g: continue
        n += 1
        t_q,_ = top1(s.get("text_candidates"))
        k_q,_ = top1(s.get("kg_candidates"))
        # fusion = normalized sum on union
        tmap = normalize(s.get("text_candidates"))
        kmap = normalize(s.get("kg_candidates"))
        union = set(tmap.keys()) | set(kmap.keys())
        fused = [(q, tmap.get(q,0.0)+kmap.get(q,0.0)) for q in union]
        fused.sort(key=lambda x: x[1], reverse=True)
        f_q = fused[0][0] if fused else None

        text_hit = (t_q==g)
        kg_hit   = (k_q==g)
        fusion_hit = (f_q==g)
        oracle_hit = (text_hit or kg_hit)

        text_ok += 1 if text_hit else 0
        kg_ok   += 1 if kg_hit else 0
        fusion_ok += 1 if fusion_hit else 0
        oracle_ok += 1 if oracle_hit else 0
        if fusion_hit and (not text_hit) and (not kg_hit): ci_count += 1

    def rate(x): return (x/n*100.0) if n else 0.0
    best_single = max(text_ok, kg_ok)
    gap = (oracle_ok - best_single)
    closure = ( (fusion_ok - best_single) / gap*100.0 ) if gap>0 else 0.0

    print(f"KG-QA eval | N={n}")
    print(f"  text@1   = {rate(text_ok):.1f}%")
    print(f"  kg@1     = {rate(kg_ok):.1f}%")
    print(f"  fusion@1 = {rate(fusion_ok):.1f}%")
    print(f"  oracle@1 = {rate(oracle_ok):.1f}%")
    print(f"  gap_closure = {closure:.1f}%")
    print(f"  CI (fusion-only wins) = {ci_count/n*100.0:.1f}%")
if __name__ == "__main__":
    main()
