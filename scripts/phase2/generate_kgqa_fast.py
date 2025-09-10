#!/usr/bin/env python3
"""
Generate KG-QA examples (2–3 hop motifs) robustly from Wikidata5M.
- Handles entities.{json|jsonl}[.gz]
- Stops cleanly if label coverage is low
- Prints coverage/intersection diagnostics
"""

import json, gzip, argparse, random, os, io
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional
from tqdm import tqdm

def _is_gz(p: str) -> bool:
    return p.endswith(".gz")

def _open_text(p: str):
    return gzip.open(p, "rt", encoding="utf-8", errors="ignore") if _is_gz(p) else open(p, "r", encoding="utf-8", errors="ignore")

def load_relations(path: str) -> Dict[str, str]:
    rel = {}
    with _open_text(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rel[parts[0]] = parts[1]
    return rel

def load_entities_any(path: str) -> Dict[str, str]:
    """
    Supports:
      - JSON array: [{"entity_id":"Q42","label":"Douglas Adams"}, ...]
      - JSONL: one object per line
      - gz or plain
    Returns: {entity_id: label}
    """
    ents = {}
    with _open_text(path) as f:
        head = f.read(4096)
        f.seek(0)
        # Heuristic: JSONL usually starts with '{'
        if head.lstrip().startswith('['):
            # JSON array
            data = json.load(f)
            for obj in data:
                eid = obj.get("entity_id") or obj.get("id") or obj.get("qid")
                lab = obj.get("label") or obj.get("name") or obj.get("title")
                if eid and lab:
                    ents[eid] = lab
        else:
            # JSONL
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                eid = obj.get("entity_id") or obj.get("id") or obj.get("qid")
                lab = obj.get("label") or obj.get("name") or obj.get("title")
                if eid and lab:
                    ents[eid] = lab
    return ents

def build_graph(triples_path: str, max_lines: int) -> Tuple[Dict[str, list], int]:
    """
    Build outgoing adjacency from triples (h, r, t).
    """
    G = defaultdict(list)
    n = 0
    with _open_text(triples_path) as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            G[h].append((r, t))
            n += 1
    return G, n

def iter_two_hop_paths(G: Dict[str, list]) -> Iterable[Tuple[str, str, str, str, str]]:
    """
    Yield 2-hop (h -r1-> m -r2-> t) paths.
    """
    keys = list(G.keys())
    random.shuffle(keys)
    for h in keys:
        outs = G.get(h)
        if not outs: 
            continue
        # Randomize first hop
        outs1 = outs[:]
        random.shuffle(outs1)
        for (r1, m) in outs1:
            outs_m = G.get(m)
            if not outs_m:
                continue
            outs2 = outs_m[:]
            random.shuffle(outs2)
            for (r2, t) in outs2:
                yield (h, r1, m, r2, t)

def generate_kgqa(
    triples: str,
    relations_path: str,
    entities_path: str,
    out_file: str,
    n_samples: int = 1000,
    max_triple_lines: int = 1_000_000,
    debug: bool = False
):
    # Load
    print("Loading relations...")
    rel = load_relations(relations_path)
    print(f" relations: {len(rel):,}")

    print("Loading entities (labels)...")
    ents = load_entities_any(entities_path)
    print(f" entities: {len(ents):,}")

    print("Building graph...")
    G, used = build_graph(triples, max_triple_lines)
    print(f" graph nodes: {len(G):,} | edges used: {used:,}")

    # Quick diagnostics on coverage
    keys = set(G.keys())
    coverage = len(keys & set(ents.keys())) / max(1, len(keys))
    print(f" node-label coverage: {coverage:.3f} (|nodes∩labeled|={len(keys & set(ents.keys())):,} / |nodes|={len(keys):,})")
    if coverage < 0.05:
        print("WARNING: Very low label coverage. IDs may not match your entities file (e.g., format mismatch).")

    # Generate questions
    print("Scanning 2-hop paths...")
    qa = []
    misses_no_label = 0
    total_paths_seen = 0

    for (h, r1, m, r2, t) in tqdm(iter_two_hop_paths(G), desc="Searching two-hop paths"):
        total_paths_seen += 1

        # Need labels for start/mid/end
        if h not in ents or m not in ents or t not in ents:
            misses_no_label += 1
            if debug and (misses_no_label % 5000 == 0):
                print(f"  debug: missing label examples so far = {misses_no_label}")
            continue

        h_lab, m_lab, t_lab = ents[h], ents[m], ents[t]
        r1_lab = rel.get(r1, "related to")
        r2_lab = rel.get(r2, "associated with")

        q = f"What is the {r2_lab} of the {r1_lab} of {h_lab}?"
        a = t_lab
        qa.append({
            "id": f"kgqa_{len(qa)}",
            "question": q,
            "answer": a,
            "source": "kg-qa-generated",
            "motif": [h, r1, m, r2, t]
        })

        if len(qa) >= n_samples:
            break

    print(f"paths scanned: {total_paths_seen:,} | collected: {len(qa):,} | missing-label paths: {misses_no_label:,}")

    # If we didn’t hit target, advise next steps
    if len(qa) < n_samples:
        print(f"NOTE: Only {len(qa)} samples found (target {n_samples}).")
        print("Try one or more of: (a) increase --max_triple_lines, (b) point --entities to the JSONL from Wikidata5M,")
        print("or (c) verify IDs (triples should use the same QIDs that exist in the entities file).")

    # Save
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fo:
        for ex in qa:
            fo.write(json.dumps(ex) + "\n")
    print(f"Saved {len(qa)} KG-QA examples → {out_file}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", required=True, help="triples file (tsv with h\\tr\\tt) [.gz ok]")
    ap.add_argument("--relations", required=True, help="relation mapping tsv: rel_id \\t label")
    ap.add_argument("--entities", required=True, help="entities.{json|jsonl}[.gz] with fields {entity_id,label}")
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--max_triple_lines", type=int, default=1000000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    generate_kgqa(
        triples=args.triples,
        relations_path=args.relations,
        entities_path=args.entities,
        out_file=args.out_file,
        n_samples=args.n_samples,
        max_triple_lines=args.max_triple_lines,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
