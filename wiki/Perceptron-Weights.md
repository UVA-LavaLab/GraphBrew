# Perceptron Weights

The perceptron weights control how AdaptiveOrder selects algorithms for each community. This page explains the weight structure, tuning strategies, and how to customize the model.

## Overview

```
scripts/weights/
├── active/               # C++ reads from here (working copy)
│   ├── type_registry.json
│   ├── type_0.json
│   └── type_N.json
├── merged/               # Accumulated from all runs
└── runs/                 # Historical snapshots
```

Each JSON file contains weights for each algorithm. When AdaptiveOrder processes a community, it computes a score for each algorithm using these weights and selects the highest-scoring one.

## How Perceptron Scoring Works

Each algorithm gets a score: `bias + Σ(w_feature × feature_value) + quadratic_terms + convergence_bonus`. Highest score wins, subject to OOD guardrail (distance > 1.5 → ORIGINAL) and ORIGINAL margin fallback (best − ORIGINAL < 0.05 → keep ORIGINAL).

See [[AdaptiveOrder-ML#perceptron-scoring-diagram]] for the full scoring diagram with all 15 linear features, 3 quadratic cross-terms, and convergence bonus.

---

## Weight File Location

Weights live in `scripts/weights/active/` (see directory tree above). The auto-clustering type system groups graphs by 7 features (modularity, log_nodes, log_edges, avg_degree, degree_variance, hub_concentration, clustering_coefficient) and trains per-cluster weights.

**Loading priority:** `PERCEPTRON_WEIGHTS_FILE` env var → best matching `type_N.json` → semantic fallback → hardcoded defaults.

```bash
# Override with custom weights
export PERCEPTRON_WEIGHTS_FILE=/path/to/custom_weights.json
```

---

## File Structure

Each type file maps algorithm names to weights. Example entry:

```json
{
  "LeidenCSR": {
    "bias": 3.5,
    "w_modularity": 0.1, "w_density": 0.05, "w_degree_variance": 0.03,
    "w_hub_concentration": 0.05, "w_log_nodes": 0.02, "w_log_edges": 0.02,
    "w_clustering_coeff": 0.04, "w_avg_path_length": 0.02, "w_diameter": 0.01,
    "w_community_count": 0.03, "w_packing_factor": 0.08,
    "w_forward_edge_fraction": 0.05, "w_working_set_ratio": 0.12,
    "w_dv_x_hub": 0.15, "w_mod_x_logn": 0.06, "w_pf_x_wsr": 0.09,
    "w_fef_convergence": 0.04, "w_reorder_time": -0.0087,
    "cache_l1_impact": 0.00021, "cache_l2_impact": 0.00019,
    "cache_l3_impact": 0.00021, "cache_dram_penalty": -7.9e-05,
    "benchmark_weights": { "pr": 1.0, "bfs": 1.0, "cc": 1.0, "sssp": 1.0, "bc": 1.0, "tc": 1.0 },
    "_metadata": { "win_rate": 0.8, "avg_speedup": 1.45, "sample_count": 20 }
  },
  "_metadata": { "enhanced_features": true, "last_updated": "2026-02-02T12:00:00" }
}
```

Algorithm names match `scripts/lib/utils.py` ALGORITHMS dict.

---

## Weight Definitions

### Core Weights

| Weight | Feature | Description |
|--------|---------|-------------|
| `bias` | - | Base preference for algorithm (higher = more likely selected) |
| `w_modularity` | modularity | Leiden community quality score (0-1) |
| `w_log_nodes` | log₁₀(nodes) | Community size (vertices) |
| `w_log_edges` | log₁₀(edges) | Community size (edges) |
| `w_density` | edge density | Edges / max possible edges |
| `w_avg_degree` | avg_degree/100 | Mean vertex degree (normalized) |
| `w_degree_variance` | degree_var/100 | Degree distribution spread |
| `w_hub_concentration` | hub_conc | Edge fraction to top 10% vertices |

### Extended Graph Structure Weights

| Weight | Feature | Description |
|--------|---------|-------------|
| `w_clustering_coeff` | clustering_coefficient | Local clustering coefficient (0-1) |
| `w_avg_path_length` | avg_path_length | Average shortest path length (BFS estimate) |
| `w_diameter` | diameter_estimate | Graph diameter (BFS estimate) |
| `w_community_count` | community_count | Number of sub-communities |

### New Graph-Aware Features

These features capture additional graph structure beyond basic topology:

| Weight | Feature | Description | Source |
|--------|---------|-------------|--------|
| `w_packing_factor` | packing_factor | Ratio of avg degree to max degree; measures degree uniformity (0-1) | IISWC'18 |
| `w_forward_edge_fraction` | forward_edge_fraction | Fraction of edges going to higher-numbered vertices; measures ordering quality | GoGraph |
| `w_working_set_ratio` | log₂(working_set_ratio+1) | `graph_bytes / LLC_size`; how many times the graph overflows last-level cache | P-OPT |

**LLC Detection:** The `working_set_ratio` is computed using `GetLLCSizeBytes()`, which detects the system's L3 cache size via `sysconf(_SC_LEVEL3_CACHE_SIZE)` on Linux with a 30 MB fallback.

### Quadratic Interaction Weights

These cross-terms capture non-linear feature interactions that improve prediction:

| Weight | Interaction | Description |
|--------|-------------|-------------|
| `w_dv_x_hub` | degree_variance × hub_concentration | High-variance + hub-heavy graphs (power-law indicators) |
| `w_mod_x_logn` | modularity × log₁₀(nodes) | Large modular graphs vs small modular graphs |
| `w_pf_x_wsr` | packing_factor × log₂(working_set_ratio+1) | Uniform-degree graphs that overflow cache |

### Convergence-Aware Weight

| Weight | Feature | Description |
|--------|---------|-------------|
| `w_fef_convergence` | forward_edge_fraction | Bonus applied **only for PR and SSSP** benchmarks; captures how edge direction affects iterative convergence |

> **Note:** The convergence bonus is added in `score()` (not `scoreBase()`) and only activates when the benchmark type is `BENCH_PR` or `BENCH_SSSP`.

### Per-Benchmark Weights

| Weight | Description |
|--------|-------------|
| `benchmark_weights.pr` | Multiplier for PageRank benchmark |
| `benchmark_weights.bfs` | Multiplier for BFS benchmark |
| `benchmark_weights.cc` | Multiplier for Connected Components benchmark |
| `benchmark_weights.sssp` | Multiplier for SSSP benchmark |
| `benchmark_weights.bc` | Multiplier for Betweenness Centrality benchmark |
| `benchmark_weights.tc` | Multiplier for Triangle Counting benchmark |

**Usage:**
- When `BENCH_GENERIC` (default), multiplier = 1.0 (no adjustment)
- When a specific benchmark is passed, the score is multiplied by the corresponding weight
- This allows algorithms to score differently for different workloads

```cpp
// C++ Usage:
SelectReorderingPerceptron(features);           // BENCH_GENERIC (multiplier = 1.0)
SelectReorderingPerceptron(features, BENCH_PR); // Uses benchmark_weights.pr
SelectReorderingPerceptron(features, "bfs");    // Uses benchmark_weights.bfs
```

### Cache Impact Weights (Optional)

| Weight | Description |
|--------|-------------|
| `cache_l1_impact` | Bonus for algorithms with high L1 hit rates |
| `cache_l2_impact` | Bonus for algorithms with high L2 hit rates |
| `cache_l3_impact` | Bonus for algorithms with high L3 hit rates |
| `cache_dram_penalty` | Penalty for DRAM access (cache misses) |

### Reorder Time Weight (Optional)

| Weight | Description |
|--------|-------------|
| `w_reorder_time` | Penalty for slow reordering (typically negative, e.g., -0.0001) |

### Metadata Fields (Auto-generated)

| Field | Description |
|-------|-------------|
| `win_rate` | Fraction of benchmarks where this algorithm was best |
| `avg_speedup` | Average speedup over ORIGINAL |
| `times_best` | Number of times this algorithm was optimal |
| `sample_count` | Number of graph/benchmark samples |
| `avg_reorder_time` | Average time to reorder (seconds) |
| `avg_l1_hit_rate` | Average L1 cache hit rate (%) |
| `avg_l2_hit_rate` | Average L2 cache hit rate (%) |
| `avg_l3_hit_rate` | Average L3 cache hit rate (%) |

---

## Score Calculation

### Formula

```
final_score = (bias + Σ(w_i × feature_i) + quadratic_terms + convergence_bonus)
              × benchmark_weights[type]
```

See [[AdaptiveOrder-ML#score-calculation-c-runtime]] for the full expanded formula with all 15 linear features, 3 quadratic terms, and safety checks.

### Safety Checks

**OOD Guardrail:** If graph distance to nearest type centroid > 1.5 → return ORIGINAL.

**ORIGINAL Margin:** If best − ORIGINAL < 0.05 → keep ORIGINAL to avoid reordering overhead for marginal gains.

### Example Calculation

Community features:
```
modularity: 0.72
log_nodes: 3.5 (1000 nodes)
log_edges: 4.0 (10000 edges)
density: 0.02
avg_degree: 20 → normalized: 0.2
degree_variance: 45 → normalized: 0.45
hub_concentration: 0.55
packing_factor: 0.6
forward_edge_fraction: 0.45
working_set_ratio: 3.2 (graph is 3.2× LLC size)
```

LeidenCSR score:
```
= 0.85                          # bias
+ 0.25 × 0.72                   # modularity: +0.18
+ 0.10 × 3.5                    # log_nodes: +0.35
+ 0.10 × 4.0                    # log_edges: +0.40
+ (-0.05) × 0.02                # density: -0.001
+ 0.15 × 0.2                    # avg_degree: +0.03
+ 0.15 × 0.45                   # degree_variance: +0.0675
+ 0.25 × 0.55                   # hub_concentration: +0.1375
+ 0.10 × 0.6                    # packing_factor: +0.06
+ 0.08 × 0.45                   # forward_edge_fraction: +0.036
+ 0.12 × log₂(3.2+1)            # working_set_ratio: +0.12 × 2.07 = +0.248
+ 0.15 × (0.45 × 0.55)          # dv×hub quadratic: +0.037
+ 0.06 × (0.72 × 3.5)           # mod×logN quadratic: +0.151
+ 0.09 × (0.6 × log₂(4.2))      # pf×wsr quadratic: +0.09 × 1.24 = +0.112

≈ 2.66
```

---

## Algorithm Name Mapping

The JSON uses algorithm names, which map to IDs:

| ID | JSON Name | Description |
|----|-----------|-------------|
| 0 | ORIGINAL | No reordering |
| 1 | RANDOM | Random permutation |
| 2 | SORT | Sort by degree |
| 3 | HUBSORT | Hub sorting |
| 4 | HUBCLUSTER | Hub clustering |
| 5 | DBG | Degree-based grouping |
| 6 | HUBSORTDBG | Hub sort + DBG |
| 7 | HUBCLUSTERDBG | Hub cluster + DBG |
| 8 | RABBITORDER | Rabbit Order (has variants) |
| 9 | GORDER | Gorder |
| 10 | CORDER | Corder |
| 11 | RCM | Reverse Cuthill-McKee |
| 12 | GraphBrewOrder | GraphBrew composite (has variants) |
| 13 | MAP | Load reordering from file |
| 14 | AdaptiveOrder | This perceptron model |
| 15 | LeidenOrder | Basic Leiden ordering (via igraph) |
| 16 | LeidenDendrogram | Leiden + Dendrogram traversal (has variants) |
| 17 | LeidenCSR | Fast CSR-native Leiden (has variants) |

> **Note:** For current variant lists, see `scripts/lib/utils.py`.

---

## Tuning Strategies

### Strategy 1: Favor One Algorithm

Make LeidenCSR almost always win:

```json
{
  "LeidenCSR": {
    "bias": 1.0,
    "w_modularity": 0.0,
    "w_log_nodes": 0.0,
    "w_log_edges": 0.0,
    "w_density": 0.0,
    "w_avg_degree": 0.0,
    "w_degree_variance": 0.0,
    "w_hub_concentration": 0.0
  }
}
```

### Strategy 2: Size-Based Selection

ORIGINAL with high bias + negative `w_log_nodes`/`w_log_edges` wins for small communities; LeidenCSR with positive log weights wins for large ones.

### Strategy 3: Structure-Based Selection

HUBCLUSTERDBG with positive `w_hub_concentration`/`w_degree_variance` wins for hub-heavy graphs; RCM with negative values wins for uniform-degree graphs.

### Strategy 4: Workload-Specific

Adjust `benchmark_weights` or feature weights to favour locality-oriented algorithms for PageRank vs bandwidth-oriented algorithms for BFS.

---

## Generating Weights

```bash
# Full pipeline (recommended)
python3 scripts/graphbrew_experiment.py --full --size small

# From existing benchmark data only
python3 scripts/graphbrew_experiment.py --phase weights

# Manual edit + validate
nano scripts/weights/active/type_0.json
python3 -c "import json; json.load(open('scripts/weights/active/type_0.json'))"
```

---

## Validating Weights

```bash
# Check what AdaptiveOrder selects
./bench/bin/pr -f graph.el -s -o 14 -n 1 2>&1 | grep -A 20 "Adaptive Reordering"
```

Output shows per-community algorithm selection (features + selected algorithm for each).

```bash
# Verbose score details
./bench/bin/pr -f graph.el -s -o 14 -n 1 -v
```

---

## Default Weights

If no weights file exists, hardcoded defaults favour ORIGINAL for small communities (high bias, negative log-size weights) and LeidenCSR for large, modular, hub-heavy communities (positive `w_modularity`, `w_hub_concentration`, log-size weights).

The **bias** is the most important weight — computed as `0.5 × avg_speedup_vs_RANDOM`. Typical values: HUBSORT ~26, SORT ~9.5, LeidenCSR ~2.3, ORIGINAL 0.5 (baseline).

---

## Training Pipeline

Three main stages: **Reorder → Benchmark → Compute Weights**.

```bash
# One-command full pipeline
python3 scripts/graphbrew_experiment.py --full --size small

# Step-by-step:
python3 scripts/graphbrew_experiment.py --phase reorder --generate-maps
python3 scripts/graphbrew_experiment.py --phase benchmark --use-maps
python3 scripts/graphbrew_experiment.py --phase weights
```

The 6-phase `--train` mode fills: (1) reorder_time, (2) bias/scale, (3) cache impacts, (4) structure weights, (5) topology weights, (6) benchmark multipliers.

See [[AdaptiveOrder-ML#training-the-perceptron]] for iterative training and progression recommendations.

---

## Gradient Update Rule

Online SGD with error signal `error = (speedup - 1.0) - current_score`:

```python
lr = 0.01
for feature in all_features:  # linear + quadratic + convergence
    weights[feature] += lr * error * feature_value
for key in w_and_cache_keys:
    weights[key] *= (1.0 - 1e-4)  # L2 regularization
```

Positive error → increase weights for features that predicted success. Negative error → decrease. ORIGINAL is trained like any other algorithm, allowing the model to learn when *not reordering* is optimal.

---

## Multi-Restart Training & Benchmark Multipliers

**Multi-restart perceptrons** (5 restarts × 800 epochs per benchmark) avoid local minima by averaging weights across independent runs with different seeds. Features are z-score normalized for stable gradients.

**Regret-aware grid search** optimizes per-benchmark multipliers (`benchmark_weights`) after base weight training. For each algorithm, it tests 30 random multiplier candidates from a log-spaced grid [0.1, 10.0], keeping combinations that maximize accuracy while minimizing regret (performance loss vs optimal).

**Variant pre-collapse:** Before saving, only the highest-bias variant per base algorithm is kept (e.g., `LeidenCSR_gveopt2` beats `LeidenCSR_gve` → saved as `LeidenCSR`).

---

## Validation

Current metrics (47 graphs × 4 benchmarks = 188 predictions): **46.8% accuracy**, **2.6% median regret**, **64.9% top-2 accuracy**, **13 unique predictions**.

See [[Python-Scripts#-eval_weightspy---weight-evaluation--c-scoring-simulation]] for the `eval_weights.py` tool.

### Cross-Validation (LOGO)

Leave-One-Graph-Out CV trains on all-but-one graph, predicts the held-out graph, and repeats. An `overfitting_score > 0.2` suggests too-specialized weights — add more graphs or increase L2 regularization.

```python
from scripts.lib.weights import cross_validate_logo
result = cross_validate_logo(benchmark_results, graph_features, type_registry)
print(f"LOGO: {result['accuracy']:.1%}, Overfit: {result['overfitting_score']:.2f}")
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Algorithm X never selected | Check bias: `grep -A1 '"AlgoX"' scripts/weights/active/type_0.json` — increase bias or feature weights |
| Wrong algorithm selected | Run with `-v` to see scores, manually verify feature × weight products |
| Weights not loading | Check `ls scripts/weights/active/type_*.json` and `echo $PERCEPTRON_WEIGHTS_FILE` |

**Best practices:** Start with moderate biases (0.5–0.7), validate empirically after tuning, keep backups (`cp -r scripts/weights/ scripts/weights.backup/`).

---

## Next Steps

- [[Correlation-Analysis]] - Automatic weight generation
- [[AdaptiveOrder-ML]] - Using the perceptron model
- [[Adding-New-Algorithms]] - Add algorithms to the perceptron

---

[← Back to Home](Home) | [GraphBrewOrder →](GraphBrewOrder)
