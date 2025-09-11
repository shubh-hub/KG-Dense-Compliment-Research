#!/usr/bin/env python3
import json, argparse, re

def toks(s): 
    return set(re.sub(r"[^a-z0-9 ]+"," "," ".join(s.lower().split())).split())

def load_labels(path="data/raw/kg/wikidata5m/wikidata5m_entity.txt"):
    d={}
    with open(path,encoding="utf-8") as f:
        for ln in f:
            p=ln.rstrip("\n").split("\t")
            if len(p)>=2: d[p[0]]=p[1]
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--entity_labels", default="data/raw/kg/wikidata5m/wikidata5m_entity.txt")
    ap.add_argument("--w_text", type=float, default=0.7)
    ap.add_argument("--w_kg", type=float, default=0.1)
    ap.add_argument("--w_lex", type=float, default=0.2)
    args=ap.parse_args()
    qid2label=load_labels(args.entity_labels)

    out=open(args.out_jsonl,"w",encoding="utf-8")
    for ln in open(args.in_jsonl,encoding="utf-8"):
        s=json.loads(ln)
        q=(s.get("text") or s.get("question",""))
        qT=toks(q)
        T=s.get("text_candidates") or []
        K=s.get("kg_candidates") or []
        def norm(c):
            if not c: return {}
            m=max(x["score"] for x in c) or 1.0
            return {x["qid"]: x["score"]/m for x in c}
        tmap, kmap = norm(T), norm(K)

        # lexical overlap
        def lex(qid):
            lab = qid2label.get(qid,"").lower()
            labT=toks(lab)
            return len(qT & labT)/max(1,len(labT))

        union={}
        union={}
        all_qids = set(tmap) | set(kmap)
        for qid in all_qids:
            score = (args.w_text * tmap.get(qid, 0) + 
                     args.w_kg * kmap.get(qid, 0) + 
                     args.w_lex * lex(qid))
            union[qid] = score

        s["fused_candidates"] = [{"qid":q,"score":union[q]} for q in sorted(union,key=union.get,reverse=True)[:20]]
        out.write(json.dumps(s,ensure_ascii=False)+"\n")
    out.close()
    print("wrote:",args.out_jsonl)

if __name__=="__main__": main()
