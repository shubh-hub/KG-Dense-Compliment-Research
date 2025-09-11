#!/usr/bin/env python3
import argparse, json, gzip, re, pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
def open_any(p): 
    return gzip.open(p,"rt",encoding="utf-8",errors="ignore") if p.endswith(".gz") else open(p,"r",encoding="utf-8",errors="ignore")
def norm(t): 
    import re; t=re.sub(r"[^a-z0-9 ]+"," ",t.lower()); return re.sub(r"\s+"," ",t.strip())
U = set(x.strip() for x in open("data/phase2/U_qids_delta.txt") if x.strip())
# labels for U
qid2label={}
for ln in open("data/raw/kg/wikidata5m/wikidata5m_entity.txt", encoding="utf-8"):
    p=ln.rstrip("\n").split("\t")
    if len(p)<2: continue
    q,l=p[0],p[1]
    if q in U: qid2label[q]=l
docs=[]; qids=[]
with open("data/phase2/entity_corpus_delta.jsonl","w",encoding="utf-8") as fo, open_any("data/raw/kg/wikidata5m/wikidata5m_text.txt.gz") as fi:
    for ln in fi:
        p=ln.rstrip("\n").split("\t",1)
        if len(p)!=2: continue
        qid, desc = p
        if qid not in U: continue
        text = ((qid2label.get(qid,"")+" "+desc).strip())
        fo.write(json.dumps({"qid":qid,"text":text}, ensure_ascii=False)+"\n")
        qids.append(qid); docs.append(norm(text).split())
bm25 = BM25Okapi(docs)
pickle.dump({"bm25":bm25,"qids":qids,"qid2idx":{q:i for i,q in enumerate(qids)}, "num_docs":len(qids)}, open("data/phase2/bm25_delta.pkl","wb"))
print(f"delta BM25 docs: {len(qids)}")
