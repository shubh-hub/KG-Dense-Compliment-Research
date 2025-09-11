#!/usr/bin/env python3
import json, argparse, pickle
def gold_qid(s):
    return s.get("motif", [None]*5)[-1] if s.get("source")=="kg-qa-generated" else None
def start_qid(s):
    return s.get("motif", [None])[0] if s.get("source")=="kg-qa-generated" else None
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--bm25_pkls", default="data/phase2/bm25_U.pkl,data/phase2/bm25_delta.pkl")
    ap.add_argument("--kg_graph_json", default="data/phase2/kg_subset_graph_expanded_v2.json")
    args=ap.parse_args()
    bm25_union=set()
    for p in args.bm25_pkls.split(","):
        P=pickle.load(open(p,"rb")); bm25_union.update(P["qids"])
    G=json.load(open(args.kg_graph_json))
    G_nodes=set(G.keys()); [G_nodes.update(vs) for vs in G.values()]
    N=0; in_bm25=0; in_graph=0; start_linked=0
    for ln in open(args.in_jsonl, encoding="utf-8"):
        s=json.loads(ln); 
        if s.get("source")!="kg-qa-generated": continue
        g=gold_qid(s); 
        if not g: continue
        N+=1
        if g in bm25_union: in_bm25+=1
        if g in G_nodes: in_graph+=1
        sq=start_qid(s)
        el = s.get("el_metadata",[])
        if any(m.get("qid")==sq for m in el): start_linked+=1
    pct=lambda a,b: 100.0*a/max(1,b)
    print(f"KG-QA items: {N}")
    print(f"Gold in BM25 (union): {in_bm25}/{N} = {pct(in_bm25,N):.1f}%")
    print(f"Gold in Graph (v2):   {in_graph}/{N} = {pct(in_graph,N):.1f}%")
    print(f"Start entity linked:  {start_linked}/{N} = {pct(start_linked,N):.1f}%")
if __name__=="__main__": main()
