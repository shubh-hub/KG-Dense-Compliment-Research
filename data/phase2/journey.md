# Phase 2 Journey: Building a High-Quality Pilot Dataset

This document chronicles the journey of Phase 2, where we transformed a raw, 12,000-sample dataset into a high-quality, sliced pilot corpus ready for complementarity experiments in Phase 3.

## 1. The Starting Point: A Skewed & Sparse Dataset

We began with a 12,000-sample corpus (`combined_12k.jsonl`) and a basic data processing pipeline. Initial analysis revealed two critical blockers for our research:

1.  **Extremely Sparse Knowledge Graph**: The initial KG subset (`kg_subset_graph.json`) was constructed only from the entities directly mentioned in the corpus. This resulted in a tiny graph of just **48 nodes and 46 edges**, making multi-hop reasoning impossible.
2.  **Unhealthy Slice Distribution**: Due to the sparse KG and a nascent entity linking (EL) pipeline, the initial slice distribution was heavily skewed. The crucial **`kg` slice was less than 1%** of the data, with the `dense` slice comprising over 90%. This was insufficient to meaningfully train or evaluate a hybrid retrieval system.

Our primary goal for Phase 2 was to address these two issues head-on to produce a dataset with a robust and realistic distribution of `kg`, `bridge`, and `dense` slices.

## 2. The Journey: Iterative Refinement and Key Breakthroughs

Our path to the final dataset involved several key iterations and methodological improvements.

### Iteration 1: Expanding the Knowledge Graph

The first major breakthrough was expanding the KG. We implemented `scripts/phase2/build_kg_subset_expanded.py` to create a 2-hop subgraph around the initially linked entities. This script:

1.  Identified all QIDs linked in the corpus (`L`).
2.  Performed a 1-hop traversal from `L` to find all neighboring nodes (`N1`).
3.  Created a union set `U = L | N1`.
4.  Extracted all triples from the master Wikidata5M file where both the head and tail entities were present in `U`.

This produced `kg_subset_graph_expanded.json`, a vastly richer graph with **105,055 nodes and 544,831 edges**. This was the foundational step that made real multi-hop pathfinding possible.

### Iteration 2: Enhancing the Entity Linking Pipeline

With a richer KG, we turned our attention to improving the EL pipeline's precision and recall. We made several surgical improvements to `scripts/phase2/run_entity_linking.py`:

-   **Smarter BM25 Query**: The disambiguation query was improved to include the mention itself plus a 120-character local context window, providing a much stronger signal to the BM25 model.
-   **Alias Normalization**: We added normalization steps to remove leading articles (`a`, `an`, `the`) and standardize apostrophes, which increased the hit rate against our alias map.
-   **Fuzzy Matching Adjustment**: The fuzzy matching score cutoff was lowered from 80 to 75, improving recall for non-exact matches.

These changes resulted in a better slice distribution, with the `kg` slice increasing to a more promising **~3%**.

### Iteration 3: The Final Push - U-Scoped Resources

This was the most critical and systematic phase of our work. We realized that for the EL pipeline to be truly effective, its resources (alias map and BM25 index) needed to be built from the **entire 105k-node expanded graph (`U`)**, not just the small set of entities found in the initial pass.

Following a detailed, step-by-step plan, we created a robust and reliable data generation pipeline using dedicated scripts:

1.  **Extract U QIDs**: Created `extract_U_qids.py` to reliably export all 105,055 QIDs from the expanded graph into `U_qids.txt`.
2.  **Build U-Scoped BM25 Index**: Created `build_bm25_U.py` to build a BM25 index (`bm25_U.pkl`) and a corresponding text corpus (`entity_corpus_U.jsonl`) using the labels and descriptions for all 105k entities.
3.  **Build U-Scoped Alias Map**: Created `build_alias_map_U.py` and `merge_alias_maps.py` to generate a comprehensive alias map (`alias_map_merged.json`) covering all entities in `U`.
4.  **Re-run EL and Slicing**: With these powerful new resources, we re-ran the entity linking and slicing pipelines one last time.

## 3. The Accomplishment: A Phase 3-Ready Dataset

This final, systematic push was a resounding success. The final dataset, `annotated_12k_sliced_v5.jsonl`, now has a healthy and realistic slice distribution, perfectly suited for our Phase 3 goals.

### Final Deliverables

All final artifacts are documented in `data/phase2/MANIFEST.json`.

-   **Final Sliced Corpus**: `data/phase2/annotated_12k_sliced_v5.jsonl`
-   **Final Slice Distribution**:
    -   **`kg`**: 1,018 samples (**8.5%**)
    -   **`bridge`**: 863 samples (**7.2%**)
    -   **`dense`**: 10,119 samples (**84.3%**)
-   **Entity Linking Quality**: A sanity check (`el_sanity_report.py`) confirmed a **73.6%** link rate among all detected mentions, with a strong mean BM25 score of **36.49**.
-   **Feature Validation**: We exported router features and trained a validation `LogisticRegression` model (`train_router_validation.py`), which achieved **perfect accuracy**, confirming that our features provide a strong signal for separating the slices.

We successfully transformed a sparse, skewed dataset into a high-quality pilot corpus. Phase 2 is complete, and we are now fully prepared to begin Phase 3: training and evaluating hybrid retrieval models.
