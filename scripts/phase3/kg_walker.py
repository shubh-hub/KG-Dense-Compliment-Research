#!/usr/bin/env python3
import json, argparse, re, os
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

ID_REL_RE = re.compile(r'(P\d+)$')
ID_ENT_RE = re.compile(r'(Q\d+)$')

def canon_rel(r: str) -> str:
    if '/' in r: r = r.rsplit('/', 1)[-1]
    if ':' in r: r = r.split(':', 1)[-1]
    m = ID_REL_RE.search(r)
    return m.group(1) if m else r

def canon_ent(q: str) -> str:
    if '/' in q: q = q.rsplit('/', 1)[-1]
    if ':' in q: q = q.split(':', 1)[-1]
    m = ID_ENT_RE.search(q)
    return m.group(1) if m else q

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

def build_per_seed_structural_allowlist(sample, adj, rel_prior, k=16):
    """Union of top-k relations from each seed, weighted by global frequency."""
    seeds = sample.get("seeds", [])
    all_rels = set()
    for s in seeds:
        if s not in adj: continue
        # Weight outgoing relations by their global prior and take top-k
        rels = sorted(adj[s], key=lambda x: rel_prior.get(x[0], 0), reverse=True)
        all_rels.update(r for r, t in rels[:k])
    return all_rels

def build_question_allowlist(question, rel_map, k):
    SYNONYMS = {
        "alma mater": "educated at", "wife": "spouse", "husband": "spouse",
        "born in": "place of birth", "died in": "place of death", "capital": "capital",
        "member of": "member of", "parent company": "parent organization", "owner": "parent organization",
        "subsidiary": "subsidiary", "wrote": "author", "author": "author",
        "directed by": "director", "starring": "cast member", "cast": "cast member"
    }
    STOP = {"of","in","the","a","an","and","or","to","for","on","by","with","at","from","as"}
    
    q_lower = (question or "").lower()
    # Expand with synonyms
    for syn, expansion in SYNONYMS.items():
        if syn in q_lower:
            q_lower += " " + expansion

    qtok = set(w for w in re.sub(r"[^a-z0-9 ]+"," ",q_lower).split() if w not in STOP)
    allow = []
    for r, lab in rel_map.items():
        lab_toks = set(lab.lower().split())
        if qtok & lab_toks:
            allow.append(r)
    return set(allow[:k])

def load_relation_map(relation_txt_path):
    rel_map = {}
    with open(relation_txt_path, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("\t")
            if len(parts) == 2:
                rel_map[parts[0]] = parts[1]
    return rel_map


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

def walk(adj, rel_prior, seeds, allowlist, max_hops, beam, hop_penalty, entity_types, rel_type_priors, type_fit_penalty, type_min_compat, kge_model, kge_beta):
    # beam over frontier: (log_score, node, path) where path=[(r,next)]
    candidates = defaultdict(lambda: (-1e9, 10**9, None, None, -1e9)) # -> (score, hop, last_rel, best_seed, path_log_prior)
    for s in seeds:
        candidates[s] = (0.0, 0, None, s, 0.0) # score, hop, last_rel, best_seed, path_log_prior
    frontier = [(0.0, s, [], s, 0.0) for s in seeds] # score, node, path, best_seed, path_log_prior
    kge_telemetry = {'applied_count': 0, 'z_scores': [], 'edge_total': 0, 'edge_scored': 0}
    # Precompute logs once
    if not (0.0 < hop_penalty <= 1.0):
        raise ValueError(f"hop_penalty must be in (0,1], got {hop_penalty}")
    if not (0.0 < type_fit_penalty <= 1.0):
        raise ValueError(f"type_fit_penalty must be in (0,1], got {type_fit_penalty}")
    log_hp = np.log(hop_penalty)
    log_pen = np.log(type_fit_penalty)
    # Log priors for speed & stability
    LOG_EPS = 1e-12
    log_rel_prior = {r: np.log(max(p, LOG_EPS)) for r, p in rel_prior.items()}

    for hop in range(1, max_hops + 1):
        new_frontier = []
        if not frontier: break
        for score, curr, path, best_seed, path_log_prior in frontier:
            if curr not in adj: continue

            edges = [(r, t) for (r, t) in adj.get(curr, []) if (not allowlist or r in allowlist)]
            # Deterministic & informed expansion
            edges.sort(key=lambda rt: rel_prior.get(rt[0], 0.0), reverse=True)

            def is_compatible(r, t):
                prior_tail = rel_type_priors.get(r, {}).get('tail_types', {})
                ttypes = entity_types.get(t, [])
                if not prior_tail or not ttypes: return True
                return bool(set(ttypes) & set(prior_tail.keys()))

            compat_mask = [is_compatible(r, t) for (r, t) in edges]
            use_types = sum(compat_mask) >= type_min_compat

            # Precompute KGE z for all type-compatible edges from this node, then zero-mean
            kge_z_map = None
            if kge_beta > 0 and kge_model and kge_model.get('stats') and use_types:
                z_vals = []
                scored_indices = []
                for i, (r, t) in enumerate(edges):
                    if compat_mask[i]:
                        kge_telemetry['edge_total'] += 1
                        h_id, t_id, r_id = canon_ent(curr), canon_ent(t), canon_rel(r)
                        if h_id in kge_model['e_map'] and r_id in kge_model['r_map'] and t_id in kge_model['e_map']:
                            h_idx, r_idx, t_idx = kge_model['e_map'][h_id], kge_model['r_map'][r_id], kge_model['e_map'][t_id]
                            h_vec, r_vec, t_vec = kge_model['e_emb'][h_idx], kge_model['r_emb'][r_idx], kge_model['e_emb'][t_idx]
                            
                            kge_score = -np.linalg.norm(h_vec + r_vec - t_vec, ord=1)
                            r_stats = kge_model['stats']['per_r'].get(r_id)
                            mu = r_stats['mu'] if r_stats else kge_model['stats']['g_mu']
                            sd = r_stats['sd'] if r_stats else kge_model['stats']['g_sd']
                            
                            z = (kge_score - mu) / (sd if sd > 1e-8 else 1.0)
                            z_vals.append(np.clip(z, -2.0, 2.0))
                            scored_indices.append(i)
                            kge_telemetry['edge_scored'] += 1
                        else:
                            z_vals.append(0.0)
                
                if z_vals:
                    z_mean = float(np.mean(z_vals))
                    centered_z = [float(z - z_mean) for z in z_vals]
                    kge_z_map = {idx: z for idx, z in zip(scored_indices, centered_z)}
                    kge_telemetry['applied_count'] += len(scored_indices)
                    kge_telemetry['z_scores'].extend(centered_z)

            for i, (r, t) in enumerate(edges):
                rel_log_prior = log_rel_prior.get(r, np.log(LOG_EPS))
                new_path_log_prior = path_log_prior + rel_log_prior
                new_log_score = score + log_hp + rel_log_prior

                type_fit = 1.0 if (not use_types or compat_mask[i]) else type_fit_penalty
                if type_fit != 1.0:
                    new_log_score += log_pen

                if kge_z_map and i in kge_z_map:
                    new_log_score += kge_beta * kge_z_map[i]

                best_log_sc, _, _, _, _ = candidates[t]
                if new_log_score > best_log_sc:
                    _, best_h, _, _, _ = candidates[t]
                    candidates[t] = (new_log_score, min(best_h, hop), r, best_seed, new_path_log_prior)
                    new_frontier.append((new_log_score, t, path + [(r, t)], best_seed, new_path_log_prior))
        # beam search after expanding all nodes at this hop
        new_frontier.sort(key=lambda x: x[0], reverse=True)
        frontier = new_frontier[:beam]
    # Filter out seeds from the final candidates
    seed_qids = {s for s in seeds}
    final_candidates = {qid: {"score": sc, "hop": h, "last_rel": lr, "best_seed": bs, "path_log_prior": plp} 
                     for qid, (sc, h, lr, bs, plp) in candidates.items() if qid not in seed_qids}
    return final_candidates, kge_telemetry

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
    ap.add_argument("--allowlist_k", type=int, default=16, help="Size of question-token allowlist")
    ap.add_argument("--struct_allow_k_per_seed", type=int, default=2, help="Per-seed top-K structural relations to union")
    ap.add_argument("--allowlist_cap", type=int, default=64, help="Cap on |question ∪ structural| allowlist size")
    ap.add_argument("--type_fit_penalty", type=float, default=1.0)
    ap.add_argument("--type_min_compat", type=int, default=3, help="Require at least this many type-compatible edges to enable type penalty at a node.")
    ap.add_argument("--rel_prior_smooth", type=float, default=0.0, help="Lambda for relation prior smoothing.")
    ap.add_argument("--kge_model_path", type=str, default=None, help="Path to KGE model artifacts directory.")
    ap.add_argument("--kge_beta", type=float, default=0.0, help="Weight for the KGE prior.")
    ap.add_argument("--expect_min_total", type=int, default=0, help="Fail if total output rows < this number (catches truncated inputs).")
    ap.add_argument("--expect_min_gold", type=int, default=0, help="Fail if gold rows < this number.")
    ap.add_argument("--min_gold_kg_cov", type=float, default=0.60, help="Fail if gold rows with kg_candidates < this fraction.")
    ap.add_argument("--text_seed_topk", type=int, default=5, help="Number of text candidates to consider for seeds.")
    args = ap.parse_args()

    adj, rel_prior = load_graph(args.graph_triples, smoothing_lambda=args.rel_prior_smooth)
    rel_map = load_relation_map(args.relation_txt)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    print("Loading type priors...")
    with open('data/phase2/type/entity_types.json', 'r', encoding='utf-8') as f:
        entity_types = json.load(f)
    with open('data/phase2/type/rel_type_prior.json', 'r', encoding='utf-8') as f:
        rel_type_priors = json.load(f)

    kge_model = None
    if args.kge_model_path:
        print(f"Loading KGE model from directory: {args.kge_model_path}...")
        try:
            kge_model = {
                'e_emb': np.load(os.path.join(args.kge_model_path, 'e_emb.npy')),
                'r_emb': np.load(os.path.join(args.kge_model_path, 'r_emb.npy')),
                'e_map': json.load(open(os.path.join(args.kge_model_path, 'e_map.json'))),
                'r_map': json.load(open(os.path.join(args.kge_model_path, 'r_map.json'))),
                'stats': {},
            }
            stats_path = os.path.join(args.kge_model_path, "kge_stats.json")
            if os.path.exists(stats_path):
                S = json.load(open(stats_path))
                kge_model['stats'] = {
                    'g_mu': S["global"]["mu"],
                    'g_sd': S["global"]["sd"],
                    'per_r': S.get("per_relation", {})
                }
                print(f"Loaded KGE stats: global(mu={kge_model['stats']['g_mu']:.2f}, sd={kge_model['stats']['g_sd']:.2f}), per_rel={len(kge_model['stats']['per_r'])}")
            print("KGE model loaded successfully.")
        except FileNotFoundError as e:
            print(f"[ERROR] KGE artifact not found: {e}. Continuing without KGE model.")
            kge_model = None
    with open(args.in_jsonl, 'r', encoding='utf-8-sig') as f_in:
        total_lines = sum(1 for _ in f_in)
        f_in.seek(0)

    agg_kge_telemetry = {'applied_count': 0, 'z_scores': [], 'edge_total': 0, 'edge_scored': 0}

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
                seeds = seeds_from_el(s, adj, text_seed_topk=args.text_seed_topk)
                question_text = s.get("question") or s.get("text") or ""
                s['seeds'] = seeds  # must be set before structural allowlist
                q_allow = build_question_allowlist(question_text, rel_map, args.allowlist_k)
                s_allow = build_per_seed_structural_allowlist(s, adj, rel_prior, k=args.struct_allow_k_per_seed)
                allow_union = (q_allow | s_allow)
                # Cap allowlist to keep precision if it grows large
                if allow_union:
                    pre_size = len(allow_union)
                    allow = set(sorted(allow_union, key=lambda r: rel_prior.get(r, 0.0), reverse=True)[:args.allowlist_cap])
                    s["allowlist_rels"] = sorted(list(allow)) if allow else []
                else:
                    pre_size = 0
                    allow = None
                    s["allowlist_rels"] = []

                # Telemetry: composition
                q_only = len(q_allow - s_allow)
                s_only = len(s_allow - q_allow)
                both   = len(q_allow & s_allow)
                s["allowlist_meta"] = {
                    "enabled": bool(allow),
                    "size": len(allow) if allow else 0,
                    "pre_cap_size": pre_size,
                    "q_allow_size": len(q_allow),
                    "s_allow_size": len(s_allow),
                    "q_only": q_only, "s_only": s_only, "both": both,
                }

                if seeds:
                    cand, kge_telemetry = walk(adj, rel_prior, seeds, allow, args.max_hops, args.beam, args.hop_penalty, entity_types, rel_type_priors, args.type_fit_penalty, args.type_min_compat, kge_model, args.kge_beta)
                    agg_kge_telemetry['applied_count'] += kge_telemetry.get('applied_count', 0)
                    agg_kge_telemetry['z_scores'].extend(kge_telemetry.get('z_scores', []))
                    agg_kge_telemetry['edge_total'] += kge_telemetry.get('edge_total', 0)
                    agg_kge_telemetry['edge_scored'] += kge_telemetry.get('edge_scored', 0)
                else:
                    cand, kge_telemetry = {}, {'applied_count': 0, 'z_scores': []}
                top = sorted(cand.items(), key=lambda x: x[1]["score"], reverse=True)[:args.topk]
                s["kg_candidates"] = [{"qid": q, **v} for q,v in top]
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
        print("[INFO] Sanity check failed.")
    
    # KGE Observability
    if agg_kge_telemetry['edge_total'] > 0:
        print("\n--- KGE Telemetry ---")
        coverage = agg_kge_telemetry['edge_scored'] * 100 / agg_kge_telemetry['edge_total']
        print(f"KGE Coverage: Scored {agg_kge_telemetry['edge_scored']} / {agg_kge_telemetry['edge_total']} ({coverage:.2f}%) of type-compatible edges.")
        if agg_kge_telemetry['z_scores']:
            z_scores = np.array(agg_kge_telemetry['z_scores'])
            print(f"Z-Score Stats: mean={np.mean(z_scores):.2f}, median={np.median(z_scores):.2f}")
            hist, _ = np.histogram(z_scores, bins=[-5, -2, 0, 2, 5])
            print(f"Z-Score Hist ([-5,-2), [-2,0), [0,2), [2,5]): {hist}\n")

    print("[INFO] Sanity check passed.")
if __name__ == "__main__":
    main()
