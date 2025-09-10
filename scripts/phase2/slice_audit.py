#!/usr/bin/env python3
import json, random, argparse, statistics as st

def load(p): 
    with open(p,encoding="utf-8") as f:
        for ln in f: yield json.loads(ln)

ap = argparse.ArgumentParser()
ap.add_argument("--sliced", required=True)
ap.add_argument("--k", type=int, default=25)
args = ap.parse_args()

N=0; slices={}
lens=[]; linked=0; mentions=0
kg,bridge=[],[]
for x in load(args.sliced):
    N+=1
    s = x.get("slice","none")
    slices[s]=slices.get(s,0)+1
    text = x.get("text") or x.get("question","")
    lens.append(len(text.split()))
    em = x.get("el_metadata",[])
    mentions += len(em)
    linked  += sum(1 for m in em if m.get("qid") and m["qid"]!="NIL")
    if s=="kg": kg.append(x)
    if s=="bridge": bridge.append(x)

print(f"Total: {N:,}  | slices: {slices}")
print(f"Text len: avg={st.mean(lens):.1f}, p50={sorted(lens)[N//2]}, p90={sorted(lens)[int(0.9*N)]}")
print(f"EL link-rate: {linked}/{mentions} = {(100*linked/max(1,mentions)):.1f}%")
random.shuffle(kg); random.shuffle(bridge)
print("\n=== KG examples ===")
for x in kg[:args.k]:
    print("- ", (x.get('text') or x.get('question',''))[:140])
    print("  EL:", [m for m in x.get('el_metadata',[]) if m.get('qid') and m['qid']!='NIL'])
print("\n=== Bridge examples ===")
for x in bridge[:args.k]:
    print("- ", (x.get('text') or x.get('question',''))[:140])
    print("  EL:", [m for m in x.get('el_metadata',[]) if m.get('qid') and m['qid']!='NIL'])
