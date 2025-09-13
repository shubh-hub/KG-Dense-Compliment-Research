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
| `CI` (approx) | 0.3%      |

## Statistical Significance

A McNemar test was performed to compare the fused model against the stronger of the two single-modality baselines (KG). The test evaluates whether there is a significant difference in their classification of whether the correct answer is in the top 20 candidates.

- **Comparison**: `fused@20` vs. `kg@20`
- **Contingency Table**:
    - `n01` (Fused correct, KG incorrect): 4
    - `n10` (Fused incorrect, KG correct): 1
- **p-value**: 0.375

**Conclusion**: The result (p=0.375) is not statistically significant at the p < 0.05 level. This suggests that while the fused model shows a small improvement in recall, the difference is not large enough to be considered statistically meaningful given the sample size.
