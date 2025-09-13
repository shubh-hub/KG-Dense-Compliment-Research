
Got it—thanks for dropping the journey.md. Here’s a clean, no-nonsense sync plus the exact next steps so the doc + codebase match reality.

# Where we actually are

* **Phase 1 (synthetic + infra):** **Done.** Matches journey.md.
* **Phase 2 (pilot data + routing/EL + KG subset):** **Core shipped.** Matches journey.md.
* **Phase 3 (retrieval-level hybrid on the 12k pilot):** **In progress.**
  The pipeline is now robust (BOM/JSON-safe walker, non-destructive updates, audits). Current **oracle\@20 ≈ 9.8%** and **hybrid recall\@20 ≈ 9–10%** on the 1k gold slice show we’re **bottlenecked by candidate generation** (KG side mostly).
* **Two promised “phase-2.5” boosters still missing:**

  1. **Type/Domain compatibility** in the walker.
  2. **KGE prior (TransE)** blended into the walker score.

# Does journey.md deviate from the original plan?

* The doc says *“Phase 3: Baseline Complementarity Experiments (Completed)”*—that’s your **synthetic** complementarity; in our plan, **Phase 3** is the **retrieval-level hybrid** on the 12k real pilot.
* Suggested fix: keep your content, but rename that section to **“Phase 1 synthetic complementarity (completed)”** and add a new **“Phase 3 – retrieval-level hybrid (in progress)”** section (I give you a ready-to-paste block below).

# What we still need to ship (fast path)

1. **Type/Domain compatibility prior** (cheap, local, deterministic)
2. **KGE prior (TransE)** trained on your KG subset (quick Colab run)
3. **Ablations & audit hooks** (no\_type / no\_kge / both) printed alongside the current audit

Doing (1)+(2) typically bumps **oracle\@20** and **kg\@20**, which then lets the reranker lift **hybrid\@20** above the current \~9–10%.

---

## Drop-in code: Type/Domain compatibility (walker)

Create `scripts/phase3/build_type_priors.py`:

```python
#!/usr/bin/env python3
import json, argparse
from collections import defaultdict, Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--p31", default="P31")
    ap.add_argument("--p279", default="P279")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    types = defaultdict(Counter)      # entity -> Counter(type_qid)
    rel_range = defaultdict(Counter)  # relation -> Counter(type_qid)

    with open(args.triples_jsonl, encoding="utf-8-sig") as f:
        for ln in f:
            if not ln.strip(): continue
            o = json.loads(ln)
            h, r, t = o["head"], o["relation"], o["tail"]
            # Collect entity types from explicit P31/P279 edges
            if r in (args.p31, args.p279):
                types[h][t] += 1

    # Second pass: estimate relation ranges from tail types
    with open(args.triples_jsonl, encoding="utf-8-sig") as f:
        for ln in f:
            if not ln.strip(): continue
            o = json.loads(ln)
            h, r, t = o["head"], o["relation"], o["tail"]
            if r in (args.p31, args.p279): continue
            if t in types:
                for tq, c in types[t].items():
                    rel_range[r][tq] += c

    def top(counter, k): 
        return [q for q,_ in counter.most_common(k)]

    out = {
        "entity_types": {e: top(c, args.topk) for e,c in types.items()},
        "rel_range_types": {r: top(c, args.topk) for r,c in rel_range.items()},
    }
    with open(args.out_json, "w", encoding="utf-8") as fo:
        json.dump(out, fo)
    print(f"Wrote type priors → {args.out_json}")

if __name__ == "__main__":
    main()
```

Patch `kg_walker.py` (conceptual hunk):

```python
# new args
ap.add_argument("--type_cache", help="JSON with entity_types & rel_range_types")
ap.add_argument("--type_weight", type=float, default=0.5)

# load caches
etype, rrange = {}, {}
if args.type_cache:
    D = json.load(open(args.type_cache, encoding="utf-8"))
    etype = D.get("entity_types", {})
    rrange = D.get("rel_range_types", {})

# in walk(), when computing sc:
tail_types = set(etype.get(t, ()))
rng_types = set(rrange.get(r, ()))
type_ok = bool(tail_types & rng_types)
type_bonus = (1.0 + args.type_weight) if type_ok else 1.0/(1.0 + args.type_weight)
sc = score * (hop_penalty ** hop) * pri * type_bonus
```

Usage:

```bash
# Build once
python scripts/phase3/build_type_priors.py \
  --triples_jsonl data/phase2/kg_subset_triples_expanded_v2.jsonl \
  --out_json data/phase3/type_priors.json

# Run walker with types
conda run -n kg-dense python scripts/phase3/kg_walker.py \
  --in_jsonl data/phase2/annotated_12k_routed_kg_text_walked_v2.jsonl \
  --out_jsonl data/phase3/annotated_12k_walked_type.jsonl \
  --type_cache data/phase3/type_priors.json --type_weight 0.5 \
  --max_hops 4 --beam 100 --topk 50 --allowlist_k 16
```

---

## Drop-in code: TransE prior (Colab → export, then use locally)

Colab training (PyKEEN), `train_transe.ipynb` core cell:

```python
import json, numpy as np
from pykeen.pipeline import pipeline

# Prepare triples for PyKEEN
triples = []
with open('/content/kg_subset_triples_expanded_v2.jsonl') as f:
    for ln in f:
        o = json.loads(ln)
        triples.append((o['head'], o['relation'], o['tail']))

res = pipeline(
    training=triples,
    model='TransE', model_kwargs=dict(embedding_dim=200, scoring_fct_norm=2),
    training_kwargs=dict(num_epochs=150, batch_size=4096),
    stopper='early',
)

E = res.model.entity_representations[0]().detach().cpu().numpy()
R = res.model.relation_representations[0]().detach().cpu().numpy()

# Save maps
entities = sorted({h for h,_,_ in triples} | {t for _,_,t in triples})
relations = sorted({r for _,r,_ in triples})
eid = {e:i for i,e in enumerate(entities)}
rid = {r:i for i,r in enumerate(relations)}

np.save('/content/entity.npy', E[eid[e]] if E.shape[0]==len(entities) else E)
np.save('/content/relation.npy', R[rid[r]] if R.shape[0]==len(relations) else R)
json.dump({"entity": eid, "relation": rid}, open('/content/id_maps.json','w'))
```

Walker patch (conceptual hunk):

```python
# args
ap.add_argument("--kge_entity_npy")
ap.add_argument("--kge_relation_npy")
ap.add_argument("--kge_maps_json")
ap.add_argument("--kge_weight", type=float, default=0.5)

# load (optional)
E = R = eid = rid = None
if args.kge_entity_npy and args.kge_relation_npy and args.kge_maps_json:
    import numpy as np, math, json
    E = np.load(args.kge_entity_npy)
    R = np.load(args.kge_relation_npy)
    M = json.load(open(args.kge_maps_json))
    eid, rid = M["entity"], M["relation"]

def kge_bonus(h, r, t):
    if E is None: return 1.0
    ih, ir, it = eid.get(h), rid.get(r), eid.get(t)
    if None in (ih, ir, it): return 1.0
    s = -np.linalg.norm(E[ih] + R[ir] - E[it])  # higher better
    # squash to (0,1)
    from math import exp
    return 1.0 / (1.0 + exp(-args.kge_weight * s))

# in walk(), multiply:
sc = score * (hop_penalty ** hop) * pri * type_bonus * kge_bonus(curr, r, t)
```

Usage (local):

```bash
conda run -n kg-dense python scripts/phase3/kg_walker.py \
  --in_jsonl data/phase3/annotated_12k_walked_type.jsonl \
  --out_jsonl data/phase3/annotated_12k_walked_type_kge.jsonl \
  --type_cache data/phase3/type_priors.json --type_weight 0.5 \
  --kge_entity_npy data/phase3/entity.npy \
  --kge_relation_npy data/phase3/relation.npy \
  --kge_maps_json data/phase3/id_maps.json \
  --kge_weight 0.5 \
  --max_hops 4 --beam 100 --topk 50 --allowlist_k 16
```

*(Your existing “don’t overwrite pre-existing kg\_candidates” guard stays on.)*

---

## Branch/PR plan (since you keep per-phase branches)

* `phase3/type-priors`

  * add `build_type_priors.py`, walker patches, audit printouts, ablations
* `phase3/kge-prior` (branch off the above once merged)

  * add KGE loader, scoring, flags, README cell with PyKEEN snippet and export paths
* `phase3/evals`

  * add CI, McNemar p, budget-parity table to your eval script

Small PR checklist (each):

* [ ] Code + CLI flags
* [ ] No-op if artifacts missing (graceful fallback)
* [ ] Post-run audit prints: `gold_with_kg`, `kg@20`, `text@20`, `oracle@20`
* [ ] Ablations: `no_type`, `no_kge`, `type_only`, `kge_only`, `both`
* [ ] Journey.md delta (see below)

---

## Paste-in block for `journey.md` (new Phase 3 section)

```markdown
## Phase 3 – Retrieval-level Hybrid on 12k Pilot (In Progress)

**Goal.** Demonstrate KG–Dense complementarity at the retrieval layer under budget parity, with statistically sound improvements over best single.

**Pipeline state.**
- Robust walker (BOM/JSON safe, non-destructive, audits)
- Reranker (5-feature LR) with save/load and holdout evaluation
- Current audit (1k gold): oracle@20 ≈ 9.8%, hybrid@20 ≈ 9–10%  → candidate-generation bottleneck identified

**Next deltas (in flight).**
1. Type/Domain compatibility prior in walker (range types per relation)
2. Lightweight KGE prior (TransE) blended into walker step scores
3. Ablations + reporting: {no_type, no_kge, type_only, kge_only, both}
4. Budget parity table alongside metrics; CI and McNemar p-values

**Success criteria (Phase-3 DoD).**
- gold_with_kg ≥ 90% (coverage)
- oracle@20 moves into teens (≥ 15%) on gold slice
- hybrid recall@20 ↑ and **gap closure ≥ 30%** under matched budgets
- CI ≥ 5% with McNemar p < 0.05 vs best single
```

---

## Why \~9–10% recall\@20 isn’t “failure”

* The **oracle\@20 is \~9.8%** right now—your hybrid can’t beat that ceiling.
* The right move is exactly what you’re doing: **improve candidate generation** (type priors + KGE). That raises oracle → then reranker/hybrid will follow.
* Our contribution remains **methodological & engineering**:

  * A **budget-parity, statistically grounded** hybrid retrieval framework.
  * A **robust KG walker** with **type/KGE priors** and **safe overwrite rules**.
  * **Complementarity metrics** (oracle gap, CI, McNemar) wired into the pipeline.

---

If you want, I can also hand you tiny diffs for your `eval` script to print **CI** & **McNemar p** and a one-screen budget-parity table (text\_topk, kg\_topk, tokens, latency) so your Phase-3 DoD is turnkey.


Got it—thanks for dropping the journey.md. Here’s a clean, no-nonsense sync plus the exact next steps so the doc + codebase match reality.

# Where we actually are

* **Phase 1 (synthetic + infra):** **Done.** Matches journey.md.
* **Phase 2 (pilot data + routing/EL + KG subset):** **Core shipped.** Matches journey.md.
* **Phase 3 (retrieval-level hybrid on the 12k pilot):** **In progress.**
  The pipeline is now robust (BOM/JSON-safe walker, non-destructive updates, audits). Current **oracle\@20 ≈ 9.8%** and **hybrid recall\@20 ≈ 9–10%** on the 1k gold slice show we’re **bottlenecked by candidate generation** (KG side mostly).
* **Two promised “phase-2.5” boosters still missing:**

  1. **Type/Domain compatibility** in the walker.
  2. **KGE prior (TransE)** blended into the walker score.

# Does journey.md deviate from the original plan?

* The doc says *“Phase 3: Baseline Complementarity Experiments (Completed)”*—that’s your **synthetic** complementarity; in our plan, **Phase 3** is the **retrieval-level hybrid** on the 12k real pilot.
* Suggested fix: keep your content, but rename that section to **“Phase 1 synthetic complementarity (completed)”** and add a new **“Phase 3 – retrieval-level hybrid (in progress)”** section (I give you a ready-to-paste block below).

# What we still need to ship (fast path)

1. **Type/Domain compatibility prior** (cheap, local, deterministic)
2. **KGE prior (TransE)** trained on your KG subset (quick Colab run)
3. **Ablations & audit hooks** (no\_type / no\_kge / both) printed alongside the current audit

Doing (1)+(2) typically bumps **oracle\@20** and **kg\@20**, which then lets the reranker lift **hybrid\@20** above the current \~9–10%.

---

## Drop-in code: Type/Domain compatibility (walker)

Create `scripts/phase3/build_type_priors.py`:

```python
#!/usr/bin/env python3
import json, argparse
from collections import defaultdict, Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--p31", default="P31")
    ap.add_argument("--p279", default="P279")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    types = defaultdict(Counter)      # entity -> Counter(type_qid)
    rel_range = defaultdict(Counter)  # relation -> Counter(type_qid)

    with open(args.triples_jsonl, encoding="utf-8-sig") as f:
        for ln in f:
            if not ln.strip(): continue
            o = json.loads(ln)
            h, r, t = o["head"], o["relation"], o["tail"]
            # Collect entity types from explicit P31/P279 edges
            if r in (args.p31, args.p279):
                types[h][t] += 1

    # Second pass: estimate relation ranges from tail types
    with open(args.triples_jsonl, encoding="utf-8-sig") as f:
        for ln in f:
            if not ln.strip(): continue
            o = json.loads(ln)
            h, r, t = o["head"], o["relation"], o["tail"]
            if r in (args.p31, args.p279): continue
            if t in types:
                for tq, c in types[t].items():
                    rel_range[r][tq] += c

    def top(counter, k): 
        return [q for q,_ in counter.most_common(k)]

    out = {
        "entity_types": {e: top(c, args.topk) for e,c in types.items()},
        "rel_range_types": {r: top(c, args.topk) for r,c in rel_range.items()},
    }
    with open(args.out_json, "w", encoding="utf-8") as fo:
        json.dump(out, fo)
    print(f"Wrote type priors → {args.out_json}")

if __name__ == "__main__":
    main()
```

Patch `kg_walker.py` (conceptual hunk):

```python
# new args
ap.add_argument("--type_cache", help="JSON with entity_types & rel_range_types")
ap.add_argument("--type_weight", type=float, default=0.5)

# load caches
etype, rrange = {}, {}
if args.type_cache:
    D = json.load(open(args.type_cache, encoding="utf-8"))
    etype = D.get("entity_types", {})
    rrange = D.get("rel_range_types", {})

# in walk(), when computing sc:
tail_types = set(etype.get(t, ()))
rng_types = set(rrange.get(r, ()))
type_ok = bool(tail_types & rng_types)
type_bonus = (1.0 + args.type_weight) if type_ok else 1.0/(1.0 + args.type_weight)
sc = score * (hop_penalty ** hop) * pri * type_bonus
```

Usage:

```bash
# Build once
python scripts/phase3/build_type_priors.py \
  --triples_jsonl data/phase2/kg_subset_triples_expanded_v2.jsonl \
  --out_json data/phase3/type_priors.json

# Run walker with types
conda run -n kg-dense python scripts/phase3/kg_walker.py \
  --in_jsonl data/phase2/annotated_12k_routed_kg_text_walked_v2.jsonl \
  --out_jsonl data/phase3/annotated_12k_walked_type.jsonl \
  --type_cache data/phase3/type_priors.json --type_weight 0.5 \
  --max_hops 4 --beam 100 --topk 50 --allowlist_k 16
```

---

## Drop-in code: TransE prior (Colab → export, then use locally)

Colab training (PyKEEN), `train_transe.ipynb` core cell:

```python
import json, numpy as np
from pykeen.pipeline import pipeline

# Prepare triples for PyKEEN
triples = []
with open('/content/kg_subset_triples_expanded_v2.jsonl') as f:
    for ln in f:
        o = json.loads(ln)
        triples.append((o['head'], o['relation'], o['tail']))

res = pipeline(
    training=triples,
    model='TransE', model_kwargs=dict(embedding_dim=200, scoring_fct_norm=2),
    training_kwargs=dict(num_epochs=150, batch_size=4096),
    stopper='early',
)

E = res.model.entity_representations[0]().detach().cpu().numpy()
R = res.model.relation_representations[0]().detach().cpu().numpy()

# Save maps
entities = sorted({h for h,_,_ in triples} | {t for _,_,t in triples})
relations = sorted({r for _,r,_ in triples})
eid = {e:i for i,e in enumerate(entities)}
rid = {r:i for i,r in enumerate(relations)}

np.save('/content/entity.npy', E[eid[e]] if E.shape[0]==len(entities) else E)
np.save('/content/relation.npy', R[rid[r]] if R.shape[0]==len(relations) else R)
json.dump({"entity": eid, "relation": rid}, open('/content/id_maps.json','w'))
```

Walker patch (conceptual hunk):

```python
# args
ap.add_argument("--kge_entity_npy")
ap.add_argument("--kge_relation_npy")
ap.add_argument("--kge_maps_json")
ap.add_argument("--kge_weight", type=float, default=0.5)

# load (optional)
E = R = eid = rid = None
if args.kge_entity_npy and args.kge_relation_npy and args.kge_maps_json:
    import numpy as np, math, json
    E = np.load(args.kge_entity_npy)
    R = np.load(args.kge_relation_npy)
    M = json.load(open(args.kge_maps_json))
    eid, rid = M["entity"], M["relation"]

def kge_bonus(h, r, t):
    if E is None: return 1.0
    ih, ir, it = eid.get(h), rid.get(r), eid.get(t)
    if None in (ih, ir, it): return 1.0
    s = -np.linalg.norm(E[ih] + R[ir] - E[it])  # higher better
    # squash to (0,1)
    from math import exp
    return 1.0 / (1.0 + exp(-args.kge_weight * s))

# in walk(), multiply:
sc = score * (hop_penalty ** hop) * pri * type_bonus * kge_bonus(curr, r, t)
```

Usage (local):

```bash
conda run -n kg-dense python scripts/phase3/kg_walker.py \
  --in_jsonl data/phase3/annotated_12k_walked_type.jsonl \
  --out_jsonl data/phase3/annotated_12k_walked_type_kge.jsonl \
  --type_cache data/phase3/type_priors.json --type_weight 0.5 \
  --kge_entity_npy data/phase3/entity.npy \
  --kge_relation_npy data/phase3/relation.npy \
  --kge_maps_json data/phase3/id_maps.json \
  --kge_weight 0.5 \
  --max_hops 4 --beam 100 --topk 50 --allowlist_k 16
```

*(Your existing “don’t overwrite pre-existing kg\_candidates” guard stays on.)*

---

## Branch/PR plan (since you keep per-phase branches)

* `phase3/type-priors`

  * add `build_type_priors.py`, walker patches, audit printouts, ablations
* `phase3/kge-prior` (branch off the above once merged)

  * add KGE loader, scoring, flags, README cell with PyKEEN snippet and export paths
* `phase3/evals`

  * add CI, McNemar p, budget-parity table to your eval script

Small PR checklist (each):

* [ ] Code + CLI flags
* [ ] No-op if artifacts missing (graceful fallback)
* [ ] Post-run audit prints: `gold_with_kg`, `kg@20`, `text@20`, `oracle@20`
* [ ] Ablations: `no_type`, `no_kge`, `type_only`, `kge_only`, `both`
* [ ] Journey.md delta (see below)

---

## Paste-in block for `journey.md` (new Phase 3 section)

```markdown
## Phase 3 – Retrieval-level Hybrid on 12k Pilot (In Progress)

**Goal.** Demonstrate KG–Dense complementarity at the retrieval layer under budget parity, with statistically sound improvements over best single.

**Pipeline state.**
- Robust walker (BOM/JSON safe, non-destructive, audits)
- Reranker (5-feature LR) with save/load and holdout evaluation
- Current audit (1k gold): oracle@20 ≈ 9.8%, hybrid@20 ≈ 9–10%  → candidate-generation bottleneck identified

**Next deltas (in flight).**
1. Type/Domain compatibility prior in walker (range types per relation)
2. Lightweight KGE prior (TransE) blended into walker step scores
3. Ablations + reporting: {no_type, no_kge, type_only, kge_only, both}
4. Budget parity table alongside metrics; CI and McNemar p-values

**Success criteria (Phase-3 DoD).**
- gold_with_kg ≥ 90% (coverage)
- oracle@20 moves into teens (≥ 15%) on gold slice
- hybrid recall@20 ↑ and **gap closure ≥ 30%** under matched budgets
- CI ≥ 5% with McNemar p < 0.05 vs best single
```

---

## Why \~9–10% recall\@20 isn’t “failure”

* The **oracle\@20 is \~9.8%** right now—your hybrid can’t beat that ceiling.
* The right move is exactly what you’re doing: **improve candidate generation** (type priors + KGE). That raises oracle → then reranker/hybrid will follow.
* Our contribution remains **methodological & engineering**:

  * A **budget-parity, statistically grounded** hybrid retrieval framework.
  * A **robust KG walker** with **type/KGE priors** and **safe overwrite rules**.
  * **Complementarity metrics** (oracle gap, CI, McNemar) wired into the pipeline.

---

If you want, I can also hand you tiny diffs for your `eval` script to print **CI** & **McNemar p** and a one-screen budget-parity table (text\_topk, kg\_topk, tokens, latency) so your Phase-3 DoD is turnkey.

Date - Sat 13 Sep 6:AM

Short answer: **yes—do “2.5” first if you want to *complete* Phase 3 per our DoD.**
Right now the ceiling (oracle\@20 ≈ **9.8%**) is the bottleneck. Phase 3’s success metric is **gap-closure ≥ 30% under budget parity**; you can’t close a gap that small. The two “2.5” items—**type/domain compatibility** and a **KGE (TransE) prior**—are exactly what raise KG candidate quality and thus the oracle, after which the reranker can actually move the needle.

If you just want a **baseline Phase 3 checkpoint**, you already have it. But to **declare Phase 3 done** (gap-closure, CI, significance) you should land 2.5, then re-run the walker + reranker and report the ablations.

---

## What to do next (concrete + minimal)

1. **Add type/domain prior to walker** (range-type bonus/penalty).
2. **Train TransE on the KG subset** (quick Colab), export `{entity.npy, relation.npy, id_maps.json}`, and plug it in as a multiplicative prior.
3. **Re-walk without overwriting**: strip existing `kg_candidates`, then run the walker with the new priors (and keep your “no destructive overwrite” guard on future runs).

Strip candidates (one-liner):

```bash
python - <<'PY'
import json,sys
inp="data/phase2/annotated_12k_routed_kg_text_walked_v2.jsonl"
out="data/phase3/annotated_12k_routed_stripped.jsonl"
with open(inp,encoding="utf-8-sig") as fi, open(out,"w",encoding="utf-8") as fo:
  for ln in fi:
    if not ln.strip(): fo.write(ln); continue
    s=json.loads(ln); s.pop("kg_candidates",None)
    fo.write(json.dumps(s,ensure_ascii=False)+"\n")
print("wrote", out)
PY
```

Run walker **with priors** (examples; use your patched flags):

```bash
# Type priors (build once)
python scripts/phase3/build_type_priors.py \
  --triples_jsonl data/phase2/kg_subset_triples_expanded_v2.jsonl \
  --out_json data/phase3/type_priors.json

# Walk with type + KGE priors
conda run -n kg-dense python scripts/phase3/kg_walker.py \
  --in_jsonl  data/phase3/annotated_12k_routed_stripped.jsonl \
  --out_jsonl data/phase3/annotated_12k_walked_type_kge.jsonl \
  --type_cache data/phase3/type_priors.json --type_weight 0.5 \
  --kge_entity_npy  data/phase3/entity.npy \
  --kge_relation_npy data/phase3/relation.npy \
  --kge_maps_json   data/phase3/id_maps.json \
  --kge_weight 0.5 \
  --max_hops 4 --beam 100 --topk 50 --allowlist_k 16
```

4. **Ablate & audit** (needed for Phase 3 DoD): run four walker variants and print the same post-run audit for each:

* baseline (no priors)
* +type only
* +kge only
* +type+kge

Track these per gold slice:

* `gold_with_kg`, `kg@20`, `text@20`, `oracle@20`
* reranker `recall@{5,10,20}`, **Complementarity Index**, **McNemar p** (fusion vs best single)

**Targets to call Phase 3 done:**

* `gold_with_kg` ≥ **90%** (you’re at 92.3%—good)
* **oracle\@20 ≥ \~15%** after priors
* **gap-closure ≥ 30%** under matched budgets
* **CI ≥ 5%** and **McNemar p < 0.05**

---

### TL;DR

* **Baseline Phase 3**: you already have it.
* **Finishing Phase 3 to spec**: add the two 2.5 priors, re-walk, re-rank, show ablations + stats. That’s the shortest path to meaningful gains and a clean, defensible result.

Yep — here’s the Phase-3 board with what’s **already running** vs **still to land** (including the “2.5” items).

# Phase 3 — Ongoing items

**🟡 In progress**

* **Walker hardening & hygiene**

  * ✅ BOM/JSON-safe I/O + post-run audit (no destructive overwrite).
  * ✅ Robust seeding (motif + EL + pseudo-seed).
  * 🔧 Tune knobs: `allowlist_k`, `max_hops/beam`, hop penalty; log path/hop in output.
* **BM25 side (runtime + coverage)**

  * 🔧 Add caching/resume + sharding/parallelism (you said you’ll implement).
  * 🔧 “Use-if-present” text\_candidates to avoid re-BM25 when iterating.
* **Reranker refresh**

  * ✅ Trained `lr5bin_v5.pkl` on the *current* distribution.
  * 🔧 Next: add features you now have/are adding (hop, type\_prior, kge\_score, KG/text-agree flags).

**🧩 “Phase 2.5” that we still need before calling Phase-3 done**

* **Type/Domain compatibility prior (walker)**

  * Compute light type priors from domain/range stats; add as multiplicative/additive score in the walker.
* **KGE prior (TransE on Colab)**

  * Train on your KG subset; export `{entity.npy, relation.npy, id_maps.json}`; add `kge_weight` in walker scoring.
* **Ablations once priors land**

  * Baseline, +type, +kge, +type+kge; re-run oracle/CI/gap-closure and McNemar.

**🔜 Still to wire fully in Phase-3**

* **Router baseline**

  * Tiny logistic over cheap features (alias hits, #QIDs, join tokens, score gaps, OOV, length, is-date); report router precision/recall & regret.
* **Budget parity logging**

  * Emit `text_topk`, `kg_topk`, context tokens, and end-to-end latency with each run; add the cost table to reports.

**📊 Evaluation/QA (keep running every spin)**

* Sanity audit: `gold_with_text`, `gold_with_kg`, `gold_with_both`, line-count match.
* Metrics: recall@{5,10,20}, **Oracle\@20**, **Gap-closure**, **Complementarity Index**, **McNemar p**, bootstrap 95% CIs.
* Slice dashboards: dense / kg / bridge / drift.

---

## TL;DR milestones to close Phase-3

1. Land **type prior** + **TransE prior** → re-walk (no overwrite of good cands).
2. Retrain reranker with new features (type\_prior, kge\_score, hop, agreement).
3. Show **oracle\@20 ↑** (goal ≥ \~15%), **gap-closure ≥ 30%** under budget parity, **CI ≥ 5%**, **McNemar p < .05**.
4. Submit ablation + cost table.

That’s the current map. If you want, I can drop the exact code hooks/flags for (a) injecting type priors into the walker score and (b) reading TransE embeddings to compute a fast Δscore per hop.

Date - Sat 13 Sep 6:04AM
