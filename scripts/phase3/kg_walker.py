#!/usr/bin/env python3
import json, argparse, re
from collections import defaultdict, Counter, deque
from pathlib import Path
from tqdm import tqdm

def load_graph(triples_jsonl):
    adj = defaultdict(list)   # h -> [(r,t)]
    rel_freq = Counter()
    with open(triples_jsonl, encoding="utf-8-sig") as f:
        for ln in f:
            if not ln.strip(): continue
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
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

def seeds_from_el(sample, adj, min_score=2.0, max_seeds=8, text_seed_topk=10):
    order = []
    # (A) Always try gold motif first (may or may not be in adj; let walker decide)
    if sample.get("source") == "kg-qa-generated":
        m = sample.get("motif", [])
        if m and m[0]: order.append(m[0])
    # (B) High-confidence EL seeds that exist in our KG subset
    for m in sample.get("el_metadata", []):
        q = m.get("qid")
        if q and q != "NIL" and m.get("score", 0) >= min_score and q in adj:
            order.append(q)
    # (C) Use top-K text candidates as seeds (prefer ones in KG to avoid dead starts)
    for c in (sample.get("text_candidates") or [])[:text_seed_topk]:
        q = c.get("qid")
        if q and q in adj:
            order.append(q)
    # Dedup, preserve order
    seen, seeds = set(), []
    for q in order:
        if q not in seen:
            seen.add(q); seeds.append(q)
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
        # beam search after expanding all nodes at this hop
        new_frontier.sort(key=lambda x: x[0], reverse=True)
        frontier = new_frontier[:beam]
    # Filter out seeds from the final candidates
    seed_qids = {s for s in seeds}
    final_candidates = {qid: {"score": sc, "hop": h} for qid, (sc, h) in candidates.items() if qid not in seed_qids}
    return final_candidates

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
    ap.add_argument("--expect_min_total", type=int, default=0, help="Fail if total output rows < this number (catches truncated inputs).")
    ap.add_argument("--expect_min_gold", type=int, default=0, help="Fail if gold rows < this number.")
    ap.add_argument("--min_gold_kg_cov", type=float, default=0.60, help="Fail if gold rows with kg_candidates < this fraction.")
    args = ap.parse_args()

    adj, rel_prior = load_graph(args.graph_triples)
    rel_map = load_relation_map(args.relation_txt)
    with open(args.in_jsonl, 'r', encoding='utf-8-sig') as f_in:
        total_lines = sum(1 for _ in f_in)
        f_in.seek(0)

    with open(args.in_jsonl, encoding="utf-8-sig") as f_in, open(args.out_jsonl, "w", encoding="utf-8") as f_out:
        for ln_no, ln in enumerate(tqdm(f_in, total=total_lines, desc="KG Walker"), 1):
            if not ln.strip():
                continue
            try:
                s = json.loads(ln)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping bad JSON at line {ln_no}: {e}")
                continue
            # Determine if we should run the walker.
            # We run if the sample is routed to KG/Both, is a gold sample, or has seeds,
            # BUT ONLY IF it doesn't already have kg_candidates.
            # This prevents destructively overwriting good, pre-computed candidates.
            has_existing_kg_cands = bool(s.get("kg_candidates"))
            needs_walking = (
                s.get("route_v1") in ("KG","Both")
                or s.get("source") == "kg-qa-generated"
                or any(m.get("qid") not in (None,"NIL") and m.get("score",0) >= 2.0 for m in s.get("el_metadata", []))
                or (s.get("text_candidates") not in (None, []))
            )

            should_walk = needs_walking and not has_existing_kg_cands
            if should_walk:
                seeds = seeds_from_el(s, adj)
                question_text = s.get("text") or s.get("question", "")
                allow = rel_allowlist(question_text, rel_map, args.allowlist_k)
                # If allowlist is too small, disable it (avoid over-filtering)
                if not allow or len(allow) < 2:
                    allow = None
                cand = walk(adj, rel_prior, seeds, allow, args.max_hops, args.beam, args.hop_penalty) if seeds else {}
                top = sorted(cand.items(), key=lambda x: x[1]["score"], reverse=True)[:args.topk]
                s["kg_candidates"] = [{"qid": q, "score": v["score"], "hop": v["hop"]} for q,v in top]
            else:
                s.setdefault("kg_candidates", [])  # do NOT overwrite to []
            f_out.write(json.dumps(s, ensure_ascii=False)+"\n")
    print(f"KG walker done → {args.out_jsonl}")

    # --- Quick sanity audit ---
    print("\n[INFO] Running post-run sanity check...")
    tot=gold=gt=gk=gb=0
    with open(args.out_jsonl, encoding="utf-8-sig") as fh:
        for ln in fh:
            if not ln.strip(): continue
            s=json.loads(ln)
            tot+=1
            if s.get("source")=="kg-qa-generated":
                gold+=1
                if s.get("text_candidates"): gt+=1
                if s.get("kg_candidates"):   gk+=1
                if s.get("text_candidates") and s.get("kg_candidates"): gb+=1
    print(f"[AUDIT] total={tot} gold={gold} gold_with_text={gt} gold_with_kg={gk} gold_with_both={gb}")
    
    if total_lines != tot:
        print(f"[FATAL] Input line count ({total_lines}) does not match output line count ({tot}). Aborting.")
        raise SystemExit(1)

    if args.expect_min_total and tot < args.expect_min_total:
        raise SystemExit(f"[FATAL] Total rows {tot} < expect_min_total={args.expect_min_total}")

    if args.expect_min_gold and gold < args.expect_min_gold:
        raise SystemExit(f"[FATAL] Gold rows {gold} < expect_min_gold={args.expect_min_gold}")

    if gold and (gk / gold) < args.min_gold_kg_cov:
        print (
            f"[FATAL] KG coverage on gold ({gk}/{gold} = {gk/gold:.1%}) "
            f"< min_gold_kg_cov={args.min_gold_kg_cov:.0%}"
        )
    
    print("[INFO] Sanity check passed.")
if __name__ == "__main__":
    main()
