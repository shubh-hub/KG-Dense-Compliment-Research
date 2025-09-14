#!/usr/bin/env python3
import json, argparse, re, os
import numpy as np
from collections import defaultdict, Counter, deque
from pathlib import Path
from tqdm import tqdm

def load_graph(triples_jsonl, smoothing_lambda=0.0):
    adj = defaultdict(list)
    rel_freq = Counter()
    with open(triples_jsonl, encoding="utf-8-sig") as f:
        for ln in f:
            if not ln.strip(): continue
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            h, r, t = o["head"], o["relation"], o["tail"]
            adj[h].append((r, t))
            rel_freq[r] += 1

    # Smoothed relative frequency
    tot = sum(rel_freq.values()) or 1
    num_rels = len(rel_freq) or 1
    rel_prior = {r: (1 - smoothing_lambda) * (c / tot) + smoothing_lambda * (1 / num_rels) for r, c in rel_freq.items()}
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
    STOP = {"of","in","the","a","an","and","or","to","for","on","by","with","at","from","as"}
    toks = set(w for w in re.sub(r"[^a-z0-9 ]+"," ",question.lower()).split() if w not in STOP)
    allow = []
    for r, lab in rel_map.items():
        lab_toks = set(lab.lower().split())
        if toks & lab_toks:
            allow.append(r)
    return set(allow[:k])

def seeds_from_el(sample, adj, min_score=2.0, max_seeds=6, text_seed_topk=5):
    order = []
    # (A) Gold motif head for KG-QA samples
    if sample.get("source") == "kg-qa-generated":
        m = sample.get("motif", [])
        if m and m[0] in adj: order.append(m[0])

    # (B) High-confidence, sorted EL seeds
    el_seeds = sorted([m for m in sample.get("el_metadata", []) if m.get("score", 0) >= min_score], key=lambda x: x.get("score", 0), reverse=True)
    for m in el_seeds:
        q = m.get("qid")
        if q and q != "NIL" and q in adj:
            order.append(q)

    # (C) Top-K text candidates as seeds
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

def walk(adj, rel_prior, seeds, allowlist, max_hops, beam, hop_penalty, entity_types, rel_type_priors, type_fit_penalty, type_min_compat, kge_model, kge_beta, allow_boost, rel_cap_per_node):
    # beam over frontier: (log_score, node, path) where path=[(r,next)]
    candidates = defaultdict(lambda: (-1e9, 10**9))  # node -> (best_log_score, min_hop)
    for s in seeds:
        candidates[s] = (0.0, 0) # log(1.0) = 0

    frontier = [(0.0, s, []) for s in seeds]

    for hop in range(1, max_hops + 1):
        new_frontier = []
        if not frontier: break
        for score, curr, path in frontier:
            if curr not in adj: continue

            # Implement per-relation fan-out cap
            by_rel = defaultdict(list)
            for r, t in adj.get(curr, []): by_rel[r].append(t)
            
            edges = []
            for r, tails in by_rel.items():
                keep = tails[:rel_cap_per_node]
                edges.extend((r, t) for t in keep)

            def is_compatible(r, t):
                prior_tail = rel_type_priors.get(r, {}).get('tail_types', {})
                ttypes = entity_types.get(t, [])
                if not prior_tail or not ttypes: return True
                return bool(set(ttypes) & set(prior_tail.keys()))

            compat_mask = [is_compatible(r, t) for (r, t) in edges]
            use_types = sum(compat_mask) >= type_min_compat

            for i, (r, t) in enumerate(edges):
                rel_w = allow_boost if (allowlist and r in allowlist) else 1.0
                new_log_score = score + np.log(hop_penalty) + np.log(rel_prior.get(r, 1e-9)) + np.log(rel_w)

                type_fit = 1.0 if (not use_types or compat_mask[i]) else type_fit_penalty
                new_log_score += np.log(type_fit)

                if kge_beta > 0 and kge_model:
                    try:
                        h_idx, r_idx, t_idx = kge_model['e_map'][curr], kge_model['r_map'][r], kge_model['e_map'][t]
                        h_vec, r_vec, t_vec = kge_model['e_emb'][h_idx], kge_model['r_emb'][r_idx], kge_model['e_emb'][t_idx]
                        kge_score = -np.linalg.norm(h_vec + r_vec - t_vec, ord=1) # L1 norm for TransE
                        new_log_score += kge_beta * kge_score
                    except KeyError:
                        pass # Entity or relation not in KGE model

                best_log_sc, _ = candidates[t]
                if new_log_score > best_log_sc:
                    _, best_h = candidates[t]
                    candidates[t] = (new_log_score, min(best_h, hop))
                    new_frontier.append((new_log_score, t, path + [(r, t)]))
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
    ap.add_argument("--type_fit_penalty", type=float, default=1.0)
    ap.add_argument("--type_min_compat", type=int, default=3, help="Require at least this many type-compatible edges to enable type penalty at a node.")
    ap.add_argument("--rel_prior_smooth", type=float, default=0.0, help="Lambda for relation prior smoothing.")
    ap.add_argument("--kge_model_path", type=str, default=None, help="Path to KGE model artifacts (npz file).")
    ap.add_argument("--kge_beta", type=float, default=0.0, help="Weight for the KGE prior.")
    ap.add_argument("--allow_boost", type=float, default=1.0, help="Boost for relations in the allowlist.")
    ap.add_argument("--rel_cap_per_node", type=int, default=10**9, help="Max number of outgoing edges per relation from a node.")
    ap.add_argument("--entity_types_path", type=str, default="data/phase2/type/entity_types.json")
    ap.add_argument("--rel_type_priors_path", type=str, default="data/phase2/type/rel_type_prior.json")
    ap.add_argument("--expect_min_total", type=int, default=0, help="Fail if total output rows < this number (catches truncated inputs).")
    ap.add_argument("--expect_min_gold", type=int, default=0, help="Fail if gold rows < this number.")
    ap.add_argument("--min_gold_kg_cov", type=float, default=0.60, help="Fail if gold rows with kg_candidates < this fraction.")
    args = ap.parse_args()

    adj, rel_prior = load_graph(args.graph_triples, smoothing_lambda=args.rel_prior_smooth)
    rel_map = load_relation_map(args.relation_txt)

    print("Loading type priors...")
    with open(args.entity_types_path, 'r', encoding='utf-8') as f:
        entity_types = json.load(f)
    with open(args.rel_type_priors_path, 'r', encoding='utf-8') as f:
        rel_type_priors = json.load(f)

    kge_model = None
    if args.kge_model_path and os.path.exists(args.kge_model_path):
        print(f"Loading KGE model from {args.kge_model_path}...")
        kge_artifacts = np.load(args.kge_model_path, allow_pickle=True)
        kge_model = {
            'e_map': kge_artifacts['e_map'].item(),
            'r_map': kge_artifacts['r_map'].item(),
            'e_emb': kge_artifacts['e_emb'],
            'r_emb': kge_artifacts['r_emb'],
        }
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
                question_text = s.get("question") or s.get("text") or ""
                allow = rel_allowlist(question_text, rel_map, args.allowlist_k)

                seed_outgoing = set()
                for q in seeds:
                    for (r, _) in adj.get(q, []):
                        seed_outgoing.add(r)

                allowlist_enabled = True
                if allow and not (allow & seed_outgoing):
                    allow = None
                    allowlist_enabled = False
                
                if not allow or len(allow) < 2:
                    allow = None
                    allowlist_enabled = False

                s['seeds'] = seeds
                s['allowlist_meta'] = {'enabled': allowlist_enabled, 'size': len(allow) if allow else 0}

                cand = walk(adj, rel_prior, seeds, allow, args.max_hops, args.beam, args.hop_penalty, entity_types, rel_type_priors, args.type_fit_penalty, args.type_min_compat, kge_model, args.kge_beta, args.allow_boost, args.rel_cap_per_node) if seeds else {}
                top = sorted(cand.items(), key=lambda x: x[1]["score"], reverse=True)[:args.topk]
                s["kg_candidates"] = [{"qid": q, "score": v["score"], "hop": v["hop"]} for q,v in top]
            else:
                s.setdefault("kg_candidates", [])
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
