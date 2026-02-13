# Perceptron Weight Analysis — Deep Reasoning Report

> **Purpose**: Explain *why* each perceptron weight exists, what graph properties it
> captures, how it correlates with proven algorithm performance, and where the
> current system breaks down.

---

## 1. Architecture Summary

The AdaptiveOrder perceptron predicts which reordering algorithm will minimize
**total time** (algorithm execution + reorder cost) for a given (graph, benchmark)
pair.

```
score(algo, graph, bench) = scoreBase(algo, graph) × benchmarkMultiplier(algo, bench)
```

`scoreBase` is a linear model: `bias + Σ(w_i × feature_i)`.

There are **two weight models**:

| Model    | File                | How it works                                | Prediction quality |
|----------|---------------------|---------------------------------------------|--------------------|
| type\_17 | `weights.json`      | Cross-benchmark averaged, single scoreBase per algorithm + bench multipliers | **4.7% accuracy — BROKEN** (always predicts SORT) |
| type\_0  | `{bench}.json`      | Per-benchmark separate perceptrons, denormalized | **46.0% accuracy** vs total-time oracle |

**The type\_17 model is a degenerate classifier.** Averaging per-benchmark
perceptrons and denormalizing inflates `w_log_nodes` and `w_log_edges` for SORT
to +2.34 / +2.80, creating additive constants of +14 / +19 that dominate
all other algorithms. SORT gets a score of 28–64 while ORIGINAL gets ~1.7.

---

## 2. Feature Inventory — What Each Feature Captures

### 2.1 Features that are discriminative (have proven correlations)

| Feature | What it measures | Spearman ρ with speedup | Why it matters mechanistically |
|---------|------------------|------------------------|-------------------------------|
| **degree\_variance** | Spread of the node degree distribution. High = power-law / scale-free; Low = regular / mesh | **+0.25 to +0.40** for SORT, RCM, GORDER, LeidenOrder across PR, PR_SPMV, CC_SV (p < 0.05) | High-variance graphs have hot-spot cache thrashing on hubs. Reordering consolidates hub access patterns into contiguous memory, reducing cache misses. Degree-sort-based algorithms (SORT, HUBSORT) directly address this. |
| **hub\_concentration** | Fraction of edges incident on the top-1% degree nodes | **+0.26 to +0.34** for same algorithms, same benchmarks | More concentrated hubs = more skewed access = more benefit from hub-aware placement. HUBSORT/HUBCLUSTER specifically reorder around hubs. |
| **modularity** (= clustering_coeff × 1.5, capped at 0.9) | Proxy for community structure strength | **−0.27 to −0.46** for PR, PR_SPMV, SSSP (reorder HURTS); **+0.26 to +0.35** for CC, CC_SV, BC (reorder HELPS) | This is the most benchmark-dependent feature. For iterative convergence algorithms (PR, SSSP), highly clustered graphs already have good natural locality — reordering disrupts it. For component-finding algorithms (CC, BC), community-aware reorderings (Leiden, GraphBrewOrder) group components together, enabling better sequential access. |
| **clustering\_coefficient** | Triangle density around each node | Same as modularity (ρ = ±0.27 to ±0.46) — identical because `modularity = CC × 1.5` | Redundant with above. The "modularity" estimate IS just clustering coefficient. |
| **dv\_x\_hub** (interaction) | `degree_variance × hub_concentration` | **+0.28 to +0.39** for PR, PR_SPMV, CC_SV | Captures the combined effect: graphs that are BOTH high-variance AND hub-concentrated benefit most from reordering. Stronger signal than either alone. |

### 2.2 Features with zero discriminative power (constant across all graphs)

| Feature | Status | Why |
|---------|--------|-----|
| **packing\_factor** | **Always 0** | Not computed by C++ feature extraction. The IISWC'18 packing factor requires a locality analysis pass that doesn't exist. Weight is trained on zero data → meaningless. |
| **forward\_edge\_fraction** | **Always 0.5 (default)** | Same — GoGraph's forward edge fraction metric not implemented. All graphs use the default value, so the weight acts as a constant offset, not a discriminator. |
| **working\_set\_ratio** | **Always 0** | P-OPT working set analysis not implemented. Dead feature. |
| **pf\_x\_wsr** (interaction) | **Always 0** | Product of two dead features. |
| **avg\_path\_length** | **Always 0** | C++ doesn't compute graph diameter/BFS at feature extraction time. |
| **diameter** | **Always 0** | Same as above. |
| **community\_count** | **Always 0** | C++ doesn't run community detection at feature extraction time (only features.json has it from Python). |

**6 of 17 features (35%) are dead.** They contribute nothing to discrimination
but their weights absorb noise during training.

### 2.3 Features with low/no correlation

| Feature | Notes |
|---------|-------|
| **log\_nodes**, **log\_edges** | After Z-score normalization, these capture graph scale. No consistent correlation with which algorithm wins — reordering benefit doesn't scale monotonically with graph size. In the denormalized type\_17 model, their weights are numerically massive (~2.34, ~2.80 for SORT) and act as algorithm-specific constant offsets rather than meaningful scale predictors. |
| **density** | `avg_degree / (nodes − 1)`. Extremely small for large sparse graphs (≈1e-5). No significant correlation found. |
| **avg\_degree / 100** | Constant (≈0.09) across all benchmark graphs since they all have similar average degree. Not discriminative in this dataset. |
| **mod\_x\_logN** (interaction) | `modularity × log(nodes)`. Correlates weakly because modularity already carries the signal. The logN factor doesn't add information. |

---

## 3. Algorithm Mechanism → Why Features Should Predict Performance

### 3.1 Degree-based reorderings (SORT, HUBSORT, HUBCLUSTER, HUBSORTDBG, HUBCLUSTERDBG)

**Mechanism**: Sort nodes by degree (descending or within clusters). Hub nodes get
low IDs → contiguous in memory → CSR row indices for hubs are adjacent.

**Why `degree_variance` and `hub_concentration` predict**: In a graph with high
degree variance, a few hub nodes dominate edge access. Without reordering, these
hubs have scattered IDs causing random cache access patterns. Degree sorting
places them consecutively, converting random access into streaming. The empirical
data confirms: ρ(degree\_variance, SORT speedup over ORIGINAL) = +0.29 on PR
(p=0.016).

**Why `modularity` anti-correlates for PR**: PageRank iterates over ALL edges
every iteration — it's a full-graph pass. On a clustered graph, the original
numbering often already groups cluster members (from how the graph was discovered
or stored). Reordering by degree BREAKS this cluster locality, hurting PR. 
Confirmed: ρ(modularity, SORT speedup on PR) = insignificant, but ρ(modularity,
GORDER speedup on PR) = −0.34 (p=0.005).

### 3.2 Community-aware reorderings (LeidenOrder, GraphBrewOrder\_leiden)

**Mechanism**: Detect communities via Leiden algorithm, then renumber nodes so that
each community is contiguous in memory.

**Why `modularity` HELPS for CC/BC**: Connected components and betweenness
centrality traverse local neighborhoods. Community-aware ordering ensures that each
component's nodes are consecutive → BFS/DFS within a component is sequential.
Confirmed: ρ(modularity, GraphBrewOrder speedup on CC) = +0.35 (p=0.003).

**Why `modularity` HURTS for PR**: PageRank needs all edges, not just within-community
edges. Community ordering optimizes intra-community access but makes inter-community
edges (crucial for convergence) more random. Data shows: ρ(modularity, LeidenOrder
speedup on PR) = −0.43 (p=0.0002, highly significant).

### 3.3 Cache-optimized reorderings (GORDER, CORDER, RCM)

**Mechanism**: GORDER minimizes cache misses by placing nodes that share neighbors
close in the ordering (window-based optimization). CORDER uses a community-then-sort
approach. RCM (Reverse Cuthill-McKee) minimizes matrix bandwidth.

**Why `degree_variance` predicts GORDER**: Higher degree variance means more cache
conflict potential. GORDER's window optimization specifically targets cache line
sharing — most effective when access patterns are highly non-uniform.
Confirmed: ρ(degree\_variance, GORDER speedup on PR) = +0.29 (p=0.015).

**Why `modularity` helps GORDER for BC but hurts for PR**: Benchmark-dependent.
GORDER preserves some community structure while optimizing cache — on BC (local
traversals), this helps. On PR (global sweeps), the inter-community edge penalty
still applies.

### 3.4 ORIGINAL (no reordering)

**When it wins**: When reorder cost exceeds algorithm speedup. In total-time terms,
ORIGINAL wins 62–93% of cases because reorder times (0.1–2s) exceed the algorithm
speedup. In algo-time-only terms, ORIGINAL only wins 6–40%.

---

## 4. Critical Bugs Found

### Bug 1: type\_17 averaged model is degenerate

**Cause**: Averaging Z-score-normalized perceptrons across benchmarks, then
denormalizing, produces weights where `w_log_nodes` and `w_log_edges` for SORT
become +2.34 and +2.80. Since `log10(nodes) ≈ 5–6` for all graphs, these contribute
+14 to +19 to SORT's score — dwarfing all other algorithms' scores.

**Fix**: Either (a) use only per-benchmark type\_0 models (already working at 46%),
or (b) constrain feature weights to a bounded range after denormalization.

### Bug 2: `_simulate_score()` in eval\_weights.py uses only 8 of 17 features

The evaluation function at `eval_weights.py:87` omits:
- `w_packing_factor`, `w_forward_edge_fraction`, `w_working_set_ratio`
- `w_dv_x_hub`, `w_mod_x_logn`, `w_pf_x_wsr`

This means the evaluation reports accuracy for a DIFFERENT scoring function than
what C++ or training uses. Any eval\_weights accuracy numbers are unreliable.

### Bug 3: 6 of 17 features (35%) are dead/constant

Features `packing_factor`, `forward_edge_fraction`, `working_set_ratio`,
`avg_path_length`, `diameter`, `community_count` are always 0 or default.
Their weights absorb training noise, potentially destabilizing the model.

### Bug 4: Modularity is benchmark-dependent but weights are benchmark-independent

The correlation analysis proves that `modularity` has OPPOSITE effects:
- PR/SSSP: higher modularity → reordering HURTS (ρ = −0.30 to −0.46)
- CC/BC: higher modularity → reordering HELPS (ρ = +0.26 to +0.35)

The type\_17 model forces a single `w_modularity` per algorithm across all
benchmarks. This is fundamentally wrong — the model can't learn the
benchmark-dependent interaction. type\_0 (per-benchmark) handles this correctly.

---

## 5. Per-Algorithm Weight Reasoning (type\_17)

For each algorithm's **dominant weight** in type\_17, here is whether the
learned value aligns with theory:

| Algorithm | Largest weight | Value | Theoretically correct? |
|-----------|---------------|-------|----------------------|
| SORT | `w_degree_variance` | +8.75 | **Direction correct** (high DV → SORT helps), but **magnitude broken** (denormalization artifact) |
| SORT | `w_log_nodes` | +2.34 | **Meaningless** — acts as constant offset, doesn't capture scale-dependent behavior |
| GORDER | `w_degree_variance` | +2.71 | Correct direction — GORDER benefits from high DV |
| RCM | `w_degree_variance` | +3.95 | Correct direction, but RCM actually benefits MORE from low DV (road networks). The total-time oracle conflation may be driving this. |
| GraphBrewOrder | `w_degree_variance` | +2.62 | Correct — community reordering helps power-law graphs |
| ORIGINAL | `bias` | +1.47 | This should be the HIGHEST bias (ORIGINAL should be default), but SORT's bias is +5.71 |
| RABBITORDER | `w_degree_variance` | −0.048 | **Direction correct** — Rabbit order already handles power-law well, so high DV doesn't help more |

---

## 6. Recommendation: What Good Weights Would Look Like

Based on the empirical correlation analysis, a principled weight design would be:

### For PR/PR\_SPMV benchmarks:
- `w_degree_variance > 0` for SORT, GORDER (high DV = more cache thrashing = more benefit from reordering)
- `w_modularity < 0` for ALL reordering algorithms (high clustering = natural locality, don't break it)
- `w_hub_concentration > 0` for HUBSORT, HUBCLUSTER (targeted hub optimization)
- ORIGINAL gets high bias (default choice when signals are weak)

### For CC/BC benchmarks:
- `w_modularity > 0` for LeidenOrder, GraphBrewOrder (community detection matches component structure)
- `w_degree_variance > 0` for HUBSORT, HUBCLUSTERDBG 
- ORIGINAL still gets moderate bias (often wins on total time)

### For BFS:
- No features strongly predict which algorithm wins (BFS is dominated by the start vertex and locality is highly variable)
- ORIGINAL should have highest bias

### For TC:
- ORIGINAL wins 41% (algo-time). Weak feature correlations. Hub-aware orderings (HUBCLUSTERDBG, DBG) help on some graphs.

---

## 7. Quantified Prediction Quality Summary

| Metric | type\_17 (averaged) | type\_0 (per-benchmark) | Random baseline |
|--------|---------------------|------------------------|-----------------|
| Accuracy vs algo-time oracle | 4.7% | 18.3% | ~7% (1/15 algos) |
| Accuracy vs total-time oracle | 4.7% | **46.0%** | ~7% |
| Always-ORIGINAL baseline vs total-time oracle | — | — | ~80% |
| Median regret vs total-time oracle | 29.1% | 0.0% | ~15% |
| Prediction diversity | 1 class (SORT) | 8+ classes | 15 classes |

**Key insight**: The per-benchmark model (type\_0) is legitimate and achieves 46%
accuracy with diverse predictions. The averaged model (type\_17) is unusable.
The always-ORIGINAL baseline achieves ~80% on total-time — beating both models —
because reorder cost makes ORIGINAL the practical winner most of the time.

The perceptron is most valuable when reorder cost is AMORTIZED (multiple
algorithms run on the same graph), where the algo-time-only oracle applies and
the per-benchmark model's 18.3% accuracy significantly outperforms the 7% random
baseline.
