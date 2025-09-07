# KG + Dense Vector Complementarity Research Plan (v2 - Peer Review Revised)

**Research Question**: Do Knowledge Graphs and Dense Vector representations capture complementary information that can be systematically leveraged for improved performance in knowledge-intensive NLP tasks?

**Revised Hypothesis**: KG and dense representations exhibit behavioral complementarity that can be measured through performance-grounded metrics and exploited via principled fusion architectures under budget-parity conditions.

## 🎯 **Core Contributions (Revised)**

1. **First controlled, cross-task study** with budget-parity and oracle-router bounds
2. **Principled complementarity measurement** using behavioral and representational metrics
3. **Systematic fusion architecture evaluation** with proper statistical validation
4. **Practical routing insights** for hybrid retrieval systems

## 📊 **Experimental Design (v2 - Focused Scope)**

### **v1 Tasks (6-8 weeks)**: QA + IR
- **Compositional KG-QA**: GrailQA, WebQSP, CWQ (subset)
- **Entity-heavy IR**: BEIR entity-centric subsets, NQ-open

### **v2 Extensions (if time permits)**: EL + KGC
- **Entity Linking**: KILT-EL, BLINK-style splits
- **KG Completion**: ogbl-wikikg2, CoDEx-M, temporal ICEWS

## 🔬 **Complementarity Measurement (Fixed)**

### **Primary: Behavioral Complementarity**
```python
# Complementarity Index
def complementarity_index(kg_correct, dense_correct, hybrid_correct):
    """P(hybrid✔ & best-single✘)"""
    best_single = kg_correct | dense_correct
    return (hybrid_correct & ~best_single).mean()

# Oracle Gap
def oracle_gap(oracle_router_score, hybrid_score):
    """Headroom for improvement"""
    return oracle_router_score - hybrid_score

# Per-item categories
categories = {
    'kg_only': kg_correct & ~dense_correct,
    'dense_only': dense_correct & ~kg_correct, 
    'both_wrong': ~kg_correct & ~dense_correct,
    'both_right': kg_correct & dense_correct,
    'hybrid_rescue': hybrid_correct & ~(kg_correct | dense_correct)
}
```

### **Secondary: Representational Analysis**
```python
# Centered Kernel Alignment (CKA)
def compute_cka(kg_similarities, dense_similarities):
    """Compare similarity matrices, not raw embeddings"""
    return cka_score(kg_similarities, dense_similarities)

# Cross-predictability
def cross_predictability(kg_emb, dense_emb, items):
    """Linear map from one space to approximate pairwise distances"""
    kg_dists = pairwise_distances(kg_emb)
    dense_dists = pairwise_distances(dense_emb)
    
    # Train linear map: kg_dists -> dense_dists
    model = LinearRegression()
    model.fit(kg_dists.flatten().reshape(-1, 1), dense_dists.flatten())
    r2_score = model.score(kg_dists.flatten().reshape(-1, 1), dense_dists.flatten())
    
    return 1 - r2_score  # Lower predictability = higher complementarity
```

## 📈 **Revised Performance Targets**

### **Task-Specific Relative Improvements**
- **QA (EM/F1)**: +5-8% relative improvement over best single system
- **IR (NDCG@10)**: +3-5 points absolute, +10-15% relative
- **Statistical Significance**: p < 0.05 with BH-FDR correction
- **Effect Size**: Cohen's d > 0.3 (medium effect)

### **Oracle Bounds**
- Report oracle-router performance to bound achievable synergy
- Complementarity Index > 0.1 (10% of items benefit from fusion)
- Oracle Gap < 5 points (reasonable headroom)

## 🏗️ **Budget Parity Protocol**

### **Strict Resource Controls**
```python
BUDGET_PARITY_CONFIG = {
    "retrieval_depth": 100,        # Same top-k for both systems
    "context_tokens": 512,         # Same input length
    "inference_time": "matched",   # Equalize latency budgets
    "memory_usage": "tracked",     # Monitor VRAM/RAM usage
    "preprocessing": "standardized" # Same tokenization/normalization
}
```

### **Fairness Controls**
- Equalize retrieval depth (top-k)
- Match token/context budgets
- Control for compute envelopes
- Plot accuracy vs cost Pareto curves

## 🎯 **v1 Fusion Architectures (Pragmatic)**

### **Keep for v1**
```python
# 1. Router-based fusion
class LearnedRouter(nn.Module):
    def __init__(self):
        # Interpretable features: alias hits, #entities, OOV rate
        self.features = ['alias_hit_rate', 'entity_count', 'oov_rate', 'score_spread']
        self.classifier = nn.Linear(len(self.features), 2)  # KG vs Dense
    
# 2. Late fusion + reranker
class LateReranker(nn.Module):
    def __init__(self):
        self.reranker = nn.Linear(kg_dim + dense_dim + score_features, 1)
    
# 3. Gating (product-of-experts)
class ProductOfExperts(nn.Module):
    def __init__(self):
        self.kg_gate = nn.Linear(input_dim, 1)
        self.dense_gate = nn.Linear(input_dim, 1)
```

### **Defer to v2**
- Full tensor bilinear fusion (parameter-heavy, brittle)
- Complex cross-modal attention (compute-heavy)
- Hierarchical multi-stage architectures

## 📊 **Updated Datasets**

### **QA Tasks**
- **GrailQA**: Compositional generalization (new)
- **WebQSP**: Multi-hop reasoning
- **CWQ**: Complex questions (subset)
- **Time-split evaluation**: Train on T, test on T+Δ

### **IR Tasks**
- **BEIR entity-centric**: TREC-CAR, NQ-open, FiQA-2018
- **KILT tasks**: Entity-grounded retrieval
- **Cross-domain**: News→Web, Movies→Open

## 🏆 **Stronger Baselines**

### **Dense Baselines**
- **BM25**: Lexical baseline
- **ColBERTv2/SPLADE-v3**: Lexical-dense hybrids
- **E5/BGE**: Current sentence embeddings
- **DPR**: Dense passage retrieval

### **KG Baselines**
- **Text-to-SPARQL**: Constrained decoding
- **Path-based multi-hop**: Learned path scorer
- **TransE/RotatE**: Lightweight KG embeddings
- **Graph-RAG**: Verbalized paths in vector store

### **Oracle Controls**
- **Best-single**: Max of KG or Dense per item
- **Oracle-router**: Perfect routing decision
- **Upper-bound**: Human performance where available

## 📈 **Proper Statistical Validation**

### **Per-Query Paired Tests**
```python
# For binary outcomes (EM)
def mcnemar_test(kg_correct, dense_correct, hybrid_correct):
    """McNemar test for paired binary outcomes"""
    from statsmodels.stats.contingency_tables import mcnemar
    
    # Create contingency table
    table = create_mcnemar_table(kg_correct, hybrid_correct)
    result = mcnemar(table, exact=True)
    return result.pvalue, result.statistic

# For ranking metrics
def approximate_randomization_test(kg_scores, hybrid_scores, n_permutations=10000):
    """Approximate randomization test for ranking metrics"""
    observed_diff = np.mean(hybrid_scores - kg_scores)
    
    null_diffs = []
    for _ in range(n_permutations):
        # Randomly swap scores for each query
        swapped = np.where(np.random.rand(len(kg_scores)) < 0.5, 
                          kg_scores, hybrid_scores)
        null_diff = np.mean(swapped - kg_scores)
        null_diffs.append(null_diff)
    
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    return p_value

# Effect sizes
def cohens_d(group1, group2):
    """Cohen's d effect size"""
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1) + 
                         (len(group2) - 1) * np.var(group2)) / 
                        (len(group1) + len(group2) - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std
```

### **Multiple Comparison Correction**
- **BH-FDR**: Benjamini-Hochberg false discovery rate
- **Pre-registered endpoints**: Primary metrics declared upfront
- **95% Confidence Intervals**: For all headline numbers

## 🔍 **Evaluation Metrics (Fixed)**

### **QA Tasks**
- **Exact Match (EM)**: Primary metric
- **Token F1**: Secondary metric
- **Calibrated Faithfulness**: Attribution accuracy
- **Remove BLEU**: Inappropriate for short answers

### **IR Tasks**
- **NDCG@10**: Primary ranking metric
- **MRR**: Mean reciprocal rank
- **Recall@100**: Coverage metric
- **KILT-style provenance**: Citation accuracy

### **Complementarity Metrics**
- **Complementarity Index**: P(hybrid✔ & best-single✘)
- **Oracle Gap**: Oracle-router - Hybrid
- **Independence Gap**: Observed vs permutation null
- **Router Coverage vs Regret**: Precision-recall curves

## ⚠️ **Threats to Validity (New Section)**

### **Pretraining Leakage**
- Both KG and dense models trained on Wikipedia/Wikidata
- **Mitigation**: Time-split evaluation, cross-domain transfer

### **Budget Parity & Engineering Bias**
- Risk of favoring one modality through implementation choices
- **Mitigation**: Strict resource controls, ablation studies

### **Domain Specificity**
- Results may not generalize across domains
- **Mitigation**: Cross-domain evaluation (news→web, movies→open)

### **Annotation Noise**
- Especially problematic in EL/IR tasks
- **Mitigation**: Human spot-checks for top errors, inter-annotator agreement

## 🚀 **Revised Timeline (v1 Focus)**

### **Weeks 1-2: Setup & Baselines**
- Environment setup (M1 + cloud)
- Implement strong baselines (BM25, DPR, ColBERT, Text-to-SPARQL)
- Dataset preparation (GrailQA, WebQSP, BEIR subsets)

### **Weeks 3-4: Core Fusion**
- Router-based fusion with interpretable features
- Late fusion + reranker
- Product-of-experts gating

### **Weeks 5-6: Evaluation & Analysis**
- Budget parity experiments
- Proper statistical testing
- Complementarity analysis (behavioral + representational)

### **Weeks 7-8: Validation & Documentation**
- Cross-domain evaluation
- Oracle bounds analysis
- Paper writing and code release

## 📊 **Results Table Template**

| Task | Metric | KG | Dense | Hybrid | Oracle | Comp.Index | p-value | Cohen's d | 95% CI |
|------|--------|----|----- -|--------|--------|------------|---------|-----------|---------|
| GrailQA | EM | 45.2 | 52.1 | **56.8** | 62.3 | 0.12 | 0.003* | 0.34 | [54.1, 59.5] |
| WebQSP | F1 | 38.7 | 41.2 | **44.9** | 48.1 | 0.15 | 0.001* | 0.42 | [42.3, 47.5] |
| NQ-open | NDCG@10 | 42.1 | 48.3 | **51.7** | 55.2 | 0.09 | 0.012* | 0.28 | [49.2, 54.2] |

*p < 0.05 after BH-FDR correction

## 🎯 **Success Criteria (Revised)**

### **Primary Endpoints**
- **Complementarity Index > 0.1** across tasks
- **Statistical significance** (p < 0.05, BH-FDR corrected)
- **Medium effect size** (Cohen's d > 0.3)
- **Oracle gap < 5 points** (reasonable headroom)

### **Secondary Endpoints**
- **Cross-domain transfer** maintains gains
- **Budget parity** controls show fair comparison
- **Router interpretability** via feature analysis
- **Representational complementarity** (CKA < 0.6)

This revised plan addresses all major peer review concerns while maintaining scientific rigor and practical feasibility within your resource constraints.
