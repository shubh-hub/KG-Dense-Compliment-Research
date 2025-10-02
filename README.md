# KG + Dense Vector Complementarity

*A reproducible study of when and how Knowledge Graphs (KG) and dense vectors work **better together** for knowledge‑intensive tasks.*

> **TL;DR**: We built a budget‑parity hybrid retrieval pipeline with a hardened KG walker, a guarded reranker, and principled complementarity metrics. After Phase 2.5 improvements, **KG@20 = 13.6%**, **Fused@20 = 13.8%**, **MRR@20 = 0.0620**, **NDCG@20 = 0.0794** on the 1k gold slice—beating either modality alone, while preserving a KG recall floor via a monotone guard.

---

## Highlights

* **Budget‑parity** evaluation with oracle and router bounds
* **Robust KG walker** with allowlists, type priors, and optional KGE prior
* **Guarded late fusion** that never drops KG hits from the final top‑K
* **Complementarity metrics** (Complementarity Index, Oracle Gap, McNemar p)
* **Reproducibility**: manifests, checksums, and one‑command eval scripts

---
# Hybrid Retrieval with Knowledge Graphs and Dense Vectors under Budget Parity

## 📖 Research Summary

This study investigates the **complementary evidence** captured by knowledge graphs (KG) and dense vector retrieval for document search. We introduce a robust **hybrid pipeline** and a budget-parity evaluation harness. The pipeline involves (i) walking a typed subgraph of **Wikidata5M**, (ii) retrieving text via **BM25/dense vectors**, and (iii) fusing candidates using a learned reranker with an explicit **Monotone Fusion Guard**. This guard is *provably* designed to preserve KG recall ($\text{KG@}k$).

Methodologically, complementarity is operationalized using **oracle bounds**, **gap-closure**, **McNemar tests**, and a **Complementarity Index (CI)**. On a 12k pilot dataset (1k gold items), the guarded fusion significantly improved performance: **MRR@20** rose from $0.017$ (KG) to $0.062$ (Hybrid), and **NDCG@20** from $0.042$ to $0.079$. This achieved a gap-closure of $\approx 18\%$ under equal top-$k$ and latency. Feature inspection revealed interpretable signals, such as a strong positive weight for the feature *in\_kg* and penalties for deep hops. We also designed and calibrated two retrieval priors—a **type/domain compatibility prior** ($\tau$-sweep) and a **KGE prior**—addressing a negative result related to global TransE score normalization via a fix using *per-relation normalization* and *type-gating*. This work provides a reproducible, statistically grounded framework and concrete engineering lessons for building hybrid retrieval systems.

## ✨ Why This Is Interesting

* **Principled Measurement:** The combination of Oracle Gap, Complementarity Index (CI), and paired significance tests (McNemar) transforms the claim "fusion helps" into a **falsifiable and statistically grounded assertion**.
* **Method–Systems Co-design:** The **Monotone Fusion Guard** enforces a critical retrieval invariant—ensuring no regressions on KG recall—while simultaneously allowing for flexible **learning-to-rank (LTR)**.
* **Calibrated Priors:** This research provides clear empirical evidence for *when* **Knowledge Graph Embeddings (KGE)** are beneficial (e.g., with type-consistent edges, per-relation scaling) and *when* they are detrimental (e.g., with naive global normalization).

## 📊 Selected Results (Pilot, 1k Gold)

| Metric | KG Only | Hybrid (Guarded) | Oracle Ceiling | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Recall@20** | $13.6\%$ | $13.8\%$ | $14.7\%$ | Guard maintains KG recall; $\approx 18\%$ gap-closure. |
| **MRR@20** | $0.017$ | $0.062$ | N/A | $\approx 3.6\times$ improvement over KG. |
| **NDCG@20** | $0.042$ | $0.079$ | N/A | Significant ranking gains (approx. randomization test). |

* **Statistical Validity:** McNemar test showed **no significant regressions** vs KG on gold slice; approx. randomization test confirmed significant ranking gains.
* **Ablations:** Type prior $\tau \in \{0.3, 0.4, 0.5\}$ stabilized hops (increased $\text{hop}-2$ share). Naive global KGE normalization reduced $\text{KG@}20$ (diagnosed and corrected with per-relation, type-gated KGE).

## 🛠️ Key Contributions (What I Built)

1.  **Budget-Parity Evaluation Harness:** Includes tools for statistical rigor: **oracle bounds, Complementarity Index (CI), McNemar tests**, and **bootstrap confidence intervals (CIs)**.
2.  **Typed KG Walker:** A sophisticated walker implementing **beam search**, a **hop penalty** function, and strict **allowlists/blocklists**, integrated with a new **type/domain compatibility prior**.
3.  **Safe KGE Integration:** Implemented a mapping-safe packaging approach (JSON maps + `.npy` files), **per-relation statistics** for score normalization, and safe application via **type-gating** and **clamping** (bounded influence).
4.  **Learned Reranker + Monotone Fusion Guard:** A LTR model complemented by the explicit guard to ensure recall invariants. Includes a robust methodology for **feature-importance inspection** to maintain interpretability.

## 🚀 Future Directions

* **KGE Scoring Swap:** Replace TransE with a model like **RotatE** to better handle temporal or relational-role sensitivities via its relation-phase component.
* **Budget-Parity Optimization:** Train a lightweight **cross-encoder reranker** that is **router-triggered** (only runs on ambiguous candidate pairs) to improve ranking quality while remaining strictly within the budget-parity constraint.
* **External Validation:** Extend the evaluation to a second, distinct domain to thoroughly test the external validity and generalizability of the complementarity metrics and hybrid framework.
## Repo Status & Roadmap

**Phase 1 (synthetic+infra):** ✅ complete
**Phase 2 (pilot corpus, EL, KG subset):** ✅ complete
**Phase 2.5 (walker + guarded rerank):** ✅ complete → **current default**
**Phase 3 (full retrieval‑level hybrid on 12k pilot):** 🟡 in progress

**Current key metrics (1k gold ‘kg‑qa‑generated’ slice, k=20+20):**

| Metric   | Value      |
| -------- | ---------- |
| KG@20    | **13.6%**  |
| Fused@20 | **13.8%**  |
| MRR@20   | **0.0620** |
| NDCG@20  | **0.0794** |

**Earlier baseline (v6_textfix_r2) for comparison:** KG@20=9.6%, Text@20=1.5%, Oracle@20=10.4%, Fused@20=9.9% (gap‑closure 37.5%).

---

## Quickstart

### 1) Environment

* Local dev: **conda** env `kg-dense` (M1 okay via CPU/MPS)
* Colab training (TransE): **use “T4 GPU”** runtime
* Python ≥ 3.9, PyTorch ≥ 2.0

```bash
# clone
git clone <your-repo-url>
cd <repo>

# (optional) create env
conda env create -f envs/phase3_base_v6r2.yml -n kg-dense
conda activate kg-dense
```

### 2) Data layout (expected)

```
repo/
├─ data/
│  ├─ phase2/
│  │  ├─ kg_subset_triples_expanded_v2.jsonl
│  │  └─ ...
│  └─ phase3/
│     ├─ gold_slice.jsonl                 # 1k gold for audits
│     ├─ sweeps/                          # walker outputs
│     └─ kge/
│        └─ transE_v1/
│           ├─ e_emb.npy
│           ├─ r_emb.npy
│           ├─ e_map.json                 # {qid: idx}
│           ├─ r_map.json                 # {Pxxx or short rel: idx}
│           └─ kge_stats.json             # (global + per-relation μ,σ)
└─ scripts/phase3/
```

> If you only have PyKEEN’s `trained_model.pkl` + `entity_to_id.tsv.gz`/`relation_to_id.tsv.gz`, use our export util to write the four files above. JSON maps must contain the **same IDs** used by the walker (QIDs and short relation IDs like `P31`).

### 3) (Optional) Build type priors

```bash
python scripts/phase3/build_type_priors.py \
  --triples_jsonl data/phase2/kg_subset_triples_expanded_v2.jsonl \
  --out_json      data/phase3/type_priors.json
```

This derives lightweight **entity types** and **relation range types** from `P31/P279` edges.

### 4) (Optional) KGE stats (for per‑relation z‑score)

```bash
conda run -n kg-dense python scripts/phase3/compute_kge_stats.py \
  --bundle    data/phase3/kge/transE_v1 \
  --train_tsv data/phase3/kge/train.tsv \
  --samples   200000
```

This writes `kge_stats.json` with global and per‑relation `(μ,σ)` for TransE scores.

### 5) Run the walker

```bash
TAU=0.3                 # type penalty factor (lower = stricter)
BETA=0.05               # small KGE weight when using z-score
OUT=data/phase3/sweeps/kge/gold_type${TAU}_kge${BETA}.jsonl

conda run -n kg-dense python scripts/phase3/kg_walker.py \
  --in_jsonl    data/phase3/gold_slice.jsonl \
  --out_jsonl   "$OUT" \
  --graph_triples data/phase2/kg_subset_triples_expanded_v2.jsonl \
  --relation_txt  data/raw/kg/wikidata5m/wikidata5m_relation.txt \
  --beam 200 --max_hops 4 --hop_penalty 0.8 \
  --allowlist_k 16 --struct_allow_k_per_seed 2 --allowlist_cap 64 \
  --type_fit_penalty $TAU \
  --type_cache    data/phase3/type_priors.json \
  --kge_model_path data/phase3/kge/transE_v1 \
  --kge_beta $BETA --text_seed_topk 5
```

### 6) Analyze telemetry

```bash
python utils/analyze_walk_telemetry.py --file_path "$OUT"
```

Reports seeds vs success, allowlist usage, and hits by hop.

### 7) Rerank (guarded)

* Train/load `lr_robust_v1.pkl` (or your chosen reranker)
* Apply **monotone guard** post‑rerank so no top‑K KG hit is lost

```bash
python utils/fuse_and_rerank.py \
  --kg_candidates "$OUT" \
  --text_candidates data/phase3/text_candidates.jsonl \
  --reranker_path models/lr_robust_v1.pkl \
  --out_jsonl data/phase3/fused_results.jsonl \
  --guard_monotone true
```

### 8) One‑command evaluation (writes a JSON report)

```bash
python utils/eval_phase3.py \
  --input data/phase3/fused_results.jsonl \
  --json_out reports/phase3_eval.json
```

Emits recall@K, MRR@K, NDCG@K, Complementarity Index, Oracle Gap, McNemar p, and 95% bootstrap CIs.

---

## Design Choices

### Budget parity

We match **top‑K**, **context tokens**, and **latency envelopes** across systems. Reports include a budget table per run.

### Walker scoring

* **Relation prior** (frequency‑smoothed)
* **Type prior** (range compatibility): penalize edges whose tails violate `rel→range_types`
* **KGE prior** (TransE): optional step‑score nudging.

  * **Best practice**: apply **per‑relation z‑score**, clamp to [−2, +2], and **gate on type‑compatible edges only** to avoid exploration collapse.
  * Fallbacks if `kge_stats.json` is missing: global z‑score or disable KGE.

### Guarded late fusion

We rerank KG ∪ Text candidates with logistic/XGB features (scores, hops, lexical overlap, agreement flags, type/KGE features), then **post‑guard** to keep all top‑K KG hits.

---

## Results Snapshot

**Phase 2.5 (current default):**

* KG@20 **13.6%** → Fused@20 **13.8%**
* MRR@20 **0.0620** (vs KG‑only 0.0174)
* NDCG@20 **0.0794** (vs KG‑only 0.0417)

**Earlier baseline (v6_textfix_r2):**
Text@20 **1.5%**, KG@20 **9.6%**, Oracle@20 **10.4%**, Fused@20 **9.9%**, gap‑closure **37.5%**, CI ≈ **0.0%**

**Takeaway:** Ranking improvements are significant; bigger recall gains now depend on **candidate generation**. Type priors and properly‑scaled KGE priors raise **oracle@20**, unlocking larger hybrid gains.

---

## Reproducibility & Manifests

* Each release writes a manifest (`manifests/*.json`) with dataset paths, parameters, and SHA‑256 hashes.
* Freeze env with `conda env export -n kg-dense > envs/<tag>.yml`.

---

## Common Pitfalls & Fixes

* **`.npz` pickle mismatch** across numpy versions → prefer `e_emb.npy`, `r_emb.npy`, `e_map.json`, `r_map.json` (plain formats).
* **Global KGE z‑score hurts recall** → switch to **per‑relation** z, clamp, and **gate by type**.
* **Silent no‑output runs** → walker now audits line counts and fails fast if mismatched.

---

## Roadmap

* ✅ Phase 2.5: walker overhaul, guarded rerank
* 🟡 Phase 3: finalize per‑relation KGE gating by default; add router baseline & cost table; run ablations (`no_type`, `no_kge`, `both`) and significance
* 🔜 Cross‑domain slice & time‑split evaluation

---

## Citation

> *Preprint in preparation. Please cite this repo until the paper is available.*

```bibtex
@misc{kg-dense-complementarity,
  title        = {KG + Dense Vector Complementarity},
  author       = {Shubham et al.},
  year         = {2025},
  howpublished = {GitHub repository},
}
```

---

**Maintainer note for admissions reviewers**: This README summarizes the state of the project and links to manifests for exact reproducibility. Please see `reports/` for evaluation JSONs and `utils/` for the evaluation harness and telemetry analyzers.
