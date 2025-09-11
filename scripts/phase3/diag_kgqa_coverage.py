#!/usr/bin/env python3
import json, argparse, pickle

def gold_qid(s):
    if s.get("source")=="kg-qa-generated":
        m=s.get("motif",[])
        return m[-1] if len(m)>=5 else None
    return None

def start_qid(s):
    if s.get("source")=="kg-qa-generated":
        m=s.get("motif",[])
        return m[0] if len(m)>=5 else None
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)  # use: data/phase2/annotated_12k_routed_kg_text.jsonl
    ap.add_argument("--bm25_pkl", default="data/phase2/bm25_U.pkl")
    ap.add_argument("--kg_graph_json", default="data/phase2/kg_subset_graph_expanded.json")
    args=ap.parse_args()

    pack=pickle.load(open(args.bm25_pkl,"rb"))
    bm25_qids=set(pack["qids"])
    import json as _json
    G=_json.load(open(args.kg_graph_json))
    G_nodes=set(G.keys()) | {v for vs in G.values() for v in vs}

    N=0
    in_bm25=0
    in_graph=0
    start_linked=0
    with open(args.in_jsonl, encoding="utf-8") as f:
        for ln in f:
            s=json.loads(ln)
            if s.get("source")!="kg-qa-generated": continue
            g=gold_qid(s)
            if not g: continue
            N+=1
            if g in bm25_qids: in_bm25+=1
            if g in G_nodes: in_graph+=1
            sq=start_qid(s)
            el = s.get("el_metadata",[])
            if any(m.get("qid")==sq for m in el): start_linked+=1

    def pct(a,b): return 100.0*a/max(1,b)
    print(f"KG-QA items: {N}")
    print(f"Gold in BM25: {in_bm25}/{N} = {pct(in_bm25,N):.1f}%")
    print(f"Gold in Graph: {in_graph}/{N} = {pct(in_graph,N):.1f}%")
    print(f"Start entity linked by EL: {start_linked}/{N} = {pct(start_linked,N):.1f}%")

if __name__=="__main__":
    main()
