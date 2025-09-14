import json
import statistics as S
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.phase3.kg_walker import seeds_from_el, load_graph

adj,_ = load_graph("data/phase2/kg_subset_triples_expanded_v2.jsonl")
cnt=[]
with open("data/phase3/gold_slice.jsonl", encoding="utf-8-sig") as f:
    for ln in f:
        if not ln.strip(): continue
        s=json.loads(ln)
        cnt.append(len(seeds_from_el(s, adj)))

print("seeds: mean", S.mean(cnt), "median", S.median(cnt), "<=1:", sum(x<=1 for x in cnt))
