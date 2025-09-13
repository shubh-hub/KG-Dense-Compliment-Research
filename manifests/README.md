# Phase 3 Baseline Metrics (v6_textfix_r2)

This document records the baseline performance metrics for the `phase3_base_v6_textfix_r2` dataset, as defined in `manifests/phase3_base_v6r2.json`.

## Evaluation Parameters

- **Dataset**: `data/phase3/annotated_12k_walked_final_v6_textfix_r2.jsonl`
- **Fused Results**: `data/phase3/annotated_12k_fused_lr5bin_v6_textfix_r2_final.jsonl`
- **Budget Parity**: 
    - `text_topk`: 20
    - `kg_topk`: 20

## Performance Metrics (on `kg-qa-generated` slice, n=1000)

| Metric        | Value     |
|---------------|-----------|
| `text@20`     | 1.5%      |
| `kg@20`       | 9.6%      |
| `oracle@20`   | 10.4%     |
| `fused@20`    | 9.9%      |
| `gap_closure` | 37.5%     |
| `CI` (exact)  | 0.0%      |

### Ranking-Based Metrics

While recall@20 measures set membership, ranking metrics like Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG) evaluate the model's ability to place correct answers higher in the list. These metrics provide a more nuanced view of the reranker's contribution.

| Metric      | Fused Model | KG Baseline | Improvement |
|-------------|-------------|-------------|-------------|
| **MRR@20**  | 0.027       | 0.016       | **+74%**    |
| **NDCG@20** | 0.043       | 0.033       | **+30%**    |

**Conclusion**: The fused model demonstrates a substantial improvement in ranking quality over the KG baseline. An approximate randomization test confirms this improvement is statistically significant (p ≈ 0.0004), indicating the reranker adds real value by promoting correct candidates.

### Independence Gap

The independence gap measures the difference between the observed union recall (`oracle@20`) and the recall expected if the two retrieval systems were statistically independent. A negative gap indicates that the two systems tend to succeed on the same queries.

- **P(Text Correct)**: 1.5%
- **P(KG Correct)**: 9.6%
- **Expected Union (if independent)**: 1.5% + 9.6% - (1.5% * 9.6%) = 10.96%
- **Observed Union (`oracle@20`)**: 10.4%
- **Independence Gap**: 10.4% - 10.96% = **-0.56%**

**Conclusion**: The negative independence gap suggests a slight overlap in the retrieval successes of the text and KG systems. They are not finding entirely separate sets of correct answers, which helps explain the low exact Complementarity Index.

## Statistical Significance

A McNemar test was performed to compare the fused model against the stronger of the two single-modality baselines (KG). The test evaluates whether there is a significant difference in their classification of whether the correct answer is in the top 20 candidates.

- **Comparison**: `fused@20` vs. `kg@20`
- **Contingency Table**:
    - `n01` (Fused correct, KG incorrect): 4
    - `n10` (Fused incorrect, KG correct): 1
- **p-value**: 0.375

**Conclusion**: The result (p=0.375) is not statistically significant at the p < 0.05 level. This suggests that while the fused model shows a small improvement in recall, the difference is not large enough to be considered statistically meaningful given the sample size.

### Per-Slice Significance Analysis

To investigate whether complementarity is more pronounced on specific data slices, McNemar tests were also run on the `bridge`, `kg`, and `dense` slices individually.

| Slice    | n (samples) | n01 (Fused Wins) | n10 (KG Wins) | p-value |
|----------|-------------|------------------|---------------|---------|
| `bridge` | 863         | 1                | 0             | 1.0     |
| `kg`     | 1018        | 2                | 1             | 1.0     |
| `dense`  | 10119       | 1                | 0             | 1.0     |

**Conclusion**: No statistical significance was found on any individual slice. The number of disagreements remains too low to detect a significant effect, even on the `bridge` slice where hybrid performance is expected to be strongest. This further reinforces that the primary bottleneck is the lack of unique, correct candidates from the retrieval stage.
