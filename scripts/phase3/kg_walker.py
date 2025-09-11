#!/usr/bin/env python3
import json, argparse, re
from collections import defaultdict, Counter, deque
from pathlib import Path

def load_graph(triples_jsonl):
    adj = defaultdict(list)   # h -> [(r,t)]
    rel_freq = Counter()
    with open(triples_jsonl, encoding="utf-8") as f:
        for ln in f:
            o = json.loads(ln)
            h, r, t = o["head"], o["relation"], o["tail"]
            adj[h].append((r,t))
            rel_freq[r]+=1
    # normalize rel priors
    tot = sum(rel_freq.values()) or 1
    rel_prior = {r:(c/tot) for r,c in rel_freq.items()}
    return adj, rel_prior

def load_relation_map(relation_txt_path):
    rel_map = {}
    with open(relation_txt_path, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("\t")
            if len(parts) == 2:
                rel_map[parts[0]] = parts[1]
    return rel_map

def rel_allowlist(question, rel_map, k):
    toks = set(re.sub(r"[^a-z0-9 ]+", " ", question.lower()).split())
    allow = []
    for r, lab in rel_map.items():
        lab_toks = set(lab.lower().split())
        if toks & lab_toks:
            allow.append(r)
    return set(allow[:k])

def seeds_from_el(sample, min_score=2.0, max_seeds=5):
    seeds = [m["qid"] for m in sample.get("el_metadata", [])
             if m.get("qid") != "NIL" and m.get("score", 0) >= min_score]
    
    # For KG-QA questions, always prepend motif[0] as seed
    if sample.get("source") == "kg-qa-generated":
        motif = sample.get("motif", [])
        if motif and motif[0] not in seeds:
            seeds.insert(0, motif[0])
    
    # Pseudo-seed fallback: if no EL seeds, use top-1 BM25 candidate
    if not seeds:
        text_candidates = sample.get("text_candidates", [])
        if text_candidates:
            seeds = [text_candidates[0]["qid"]]
    
    return seeds[:max_seeds]

def walk(adj, rel_prior, seeds, allowlist, max_hops=3, beam=20, hop_penalty=0.8):
    # beam over frontier: (score, node, path) where path=[(r,next)]
    # start score = 1.0 per seed
    candidates = defaultdict(lambda: (-1.0, 10**9))  # node -> (best_score, min_hop)
    # Initialize candidates with seeds at hop 0
    for s in seeds:
        candidates[s] = (1.0, 0)

    frontier = [(1.0, s, []) for s in seeds]

    for hop in range(1, max_hops + 1):
        new_frontier = []
        if not frontier: break
        for score, curr, path in frontier:
            if curr not in adj: continue
            for r, t in adj[curr]:
                # relation filtering
                if allowlist and r not in allowlist: continue
                # update score
                pri = rel_prior.get(r, 1.0)
                sc = score * (hop_penalty ** hop) * pri
                best_sc, _ = candidates[t]
                if sc > best_sc:
                    _, best_h = candidates[t]
                    candidates[t] = (sc, min(best_h, hop))
                    new_frontier.append((sc, t, path + [(r, t)]))
            # beam search
            new_frontier.sort(key=lambda x: x[0], reverse=True)
            frontier = new_frontier[:beam]
    return {qid: {"score": sc, "hop": h} for qid, (sc, h) in candidates.items()}  # dict qid->score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_triples", default="data/phase2/kg_subset_triples_expanded_v2.jsonl")
    ap.add_argument("--relation_txt", default="data/raw/kg/wikidata5m/wikidata5m_relation.txt")
    ap.add_argument("--in_jsonl", required=True)   # routed file
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--max_hops", type=int, default=3)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--hop_penalty", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--allowlist_k", type=int, default=8, help="Size of relation allowlist")
    args = ap.parse_args()

    adj, rel_prior = load_graph(args.graph_triples)
    rel_map = load_relation_map(args.relation_txt)
    out = open(args.out_jsonl,"w",encoding="utf-8")
    with open(args.in_jsonl, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            s = json.loads(ln)
            if s.get("route_v1") in ("KG","Both"):
                seeds = seeds_from_el(s)
                question_text = s.get("text") or s.get("question", "")
                allow = rel_allowlist(question_text, rel_map, args.allowlist_k)
                cand = walk(adj, rel_prior, seeds, allow, args.max_hops, args.beam, args.hop_penalty) if seeds else {}
                # keep topk
                top = sorted(cand.items(), key=lambda x: x[1]["score"], reverse=True)[:args.topk]
                s["kg_candidates"] = [{"qid": q, "score": v["score"], "hop": v["hop"]} for q,v in top]
            out.write(json.dumps(s, ensure_ascii=False)+"\n")
    out.close()
    print(f"KG walker done → {args.out_jsonl}")
if __name__ == "__main__":
    main()
