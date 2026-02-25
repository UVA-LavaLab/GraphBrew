# Perceptron Weights

The perceptron weights control how AdaptiveOrder selects algorithms. This page explains the weight structure, tuning strategies, and how to customize the model.

## Overview

```
results/weights/
├── registry.json           # Graph→type mapping + centroids
├── type_0/
│   └── weights.json
├── type_N/
│   └── weights.json
└── ...

results/data/
└── adaptive_models.json    # Unified model store (DT, hybrid, kNN models)
```

Each JSON file contains weights for each algorithm. When AdaptiveOrder processes a graph, it computes a score for each algorithm using these weights and selects the highest-scoring one.

> **Feature vector:** The perceptron uses a 17-element feature vector with transforms (`log10`, `/100`, `/10`, `/50`, `log2`, quadratic products) matching the C++ `scoreBase()` implementation.

## How Perceptron Scoring Works

Each algorithm gets a score: `bias + Σ(w_feature × feature_value) + quadratic_terms + convergence_bonus`. Highest score wins, subject to OOD guardrail (distance > 1.5 → ORIGINAL) and ORIGINAL margin fallback (best − ORIGINAL < 0.05 → keep ORIGINAL).

See [[AdaptiveOrder-ML#perceptron-scoring-diagram]] for the full scoring diagram with all 15 linear features, 3 quadratic cross-terms, and convergence bonus.

---

## Weight File Location

Weights live in `results/weights/` (see directory tree above). The auto-clustering type system groups graphs by 7 features (modularity, log_nodes, log_edges, avg_degree, degree_variance, hub_concentration, clustering_coefficient) and trains per-cluster weights.

**Loading priority:** `PERCEPTRON_WEIGHTS_FILE` env var → per-benchmark `type_N/{bench}.json` → best matching `type_N/weights.json` → `type_0/weights.json` → hardcoded defaults.

```bash
# Override with custom weights
export PERCEPTRON_WEIGHTS_FILE=/path/to/custom_weights.json
```

---

## File Structure

Each type file maps algorithm names to weights. Example entry:

```json
{
  "ALGORITHM_NAME": {
    "bias": ...,
    "w_modularity": ..., "w_density": ..., "w_degree_variance": ...,
    "w_hub_concentration": ..., "w_log_nodes": ..., "w_log_edges": ...,
    "w_clustering_coeff": ..., "w_avg_path_length": ..., "w_diameter": ...,
    "w_community_count": ..., "w_packing_factor": ...,
    "w_forward_edge_fraction": ..., "w_working_set_ratio": ...,
    "w_dv_x_hub": ..., "w_mod_x_logn": ..., "w_pf_x_wsr": ...,
    "w_fef_convergence": ..., "w_reorder_time": ...,
    "cache_l1_impact": ..., "cache_l2_impact": ...,
    "cache_l3_impact": ..., "cache_dram_penalty": ...,
    "benchmark_weights": { "pr": 1.0, "bfs": 1.0, "cc": 1.0, "sssp": 1.0, "bc": 1.0, "tc": 1.0, "pr_spmv": 1.0, "cc_sv": 1.0 },
    "_metadata": { "win_rate": ..., "avg_speedup": ..., "sample_count": ... }
  },
  "_metadata": { "enhanced_features": true, "last_updated": "..." }
}
```

Run `--train` to generate actual weight values. See `results/weights/type_0/weights.json` for real data.

Algorithm names are produced by `canonical_algo_key(algo_id, variant)` from `scripts/lib/utils.py`. This is the SSOT for weight file keys, `.sg` filenames, and result JSON. See [[Code-Architecture#unified-naming-convention-ssot]].

---

## Weight Definitions

### Core Weights

| Weight | Feature | Description |
|--------|---------|-------------|
| `bias` | - | Base preference for algorithm (higher = more likely selected) |
| `w_modularity` | modularity | Leiden community quality score (0-1) |
| `w_log_nodes` | log₁₀(nodes+1) | Graph size (vertices) |
| `w_log_edges` | log₁₀(edges+1) | Graph size (edges) |
| `w_density` | edge density | Edges / max possible edges |
| `w_avg_degree` | avg_degree/100 | Mean vertex degree (normalized) |
| `w_degree_variance` | degree_variance | Degree distribution spread (not normalized) |
| `w_hub_concentration` | hub_conc | Edge fraction to top 10% vertices |

### Extended Graph Structure Weights

| Weight | Feature | Description |
|--------|---------|-------------|
| `w_clustering_coeff` | clustering_coefficient | Local clustering coefficient (0-1) |
| `w_avg_path_length` | avg_path_length | Average BFS distance from sampled sources |
| `w_diameter` | diameter_estimate | Max BFS depth (diameter lower bound) |
| `w_community_count` | community_count | Connected component count |

> **Note:** These features were previously always 0 at runtime. They are now computed via `ComputeExtendedFeatures()` using multi-source BFS (~50-500ms overhead).

### New Graph-Aware Features

These features capture additional graph structure beyond basic topology:

| Weight | Feature | Description | Source |
|--------|---------|-------------|--------|
| `w_packing_factor` | packing_factor | Fraction of hub neighbors already co-located (nearby vertex IDs); measures locality (0-1) | IISWC'18 |
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
| `w_fef_convergence` | forward_edge_fraction | Bonus applied **only for PR, PR_SPMV, and SSSP** benchmarks; captures how edge direction affects iterative convergence |

> **Note:** The convergence bonus is added in `score()` (not `scoreBase()`) and only activates when the benchmark type is `BENCH_PR`, `BENCH_PR_SPMV`, or `BENCH_SSSP`.

### Per-Benchmark Weights

| Weight | Description |
|--------|-------------|
| `benchmark_weights.pr` | Multiplier for PageRank benchmark |
| `benchmark_weights.bfs` | Multiplier for BFS benchmark |
| `benchmark_weights.cc` | Multiplier for Connected Components benchmark |
| `benchmark_weights.sssp` | Multiplier for SSSP benchmark |
| `benchmark_weights.bc` | Multiplier for Betweenness Centrality benchmark |
| `benchmark_weights.tc` | Multiplier for Triangle Counting benchmark |
| `benchmark_weights.pr_spmv` | Multiplier for PageRank SpMV benchmark |
| `benchmark_weights.cc_sv` | Multiplier for CC Shiloach-Vishkin benchmark |

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

The score for each algorithm is computed as:

```
score = bias
      + w_modularity × modularity
      + w_log_nodes × log₁₀(nodes+1)
      + w_log_edges × log₁₀(edges+1)
      + w_density × density
      + w_avg_degree × avg_degree / 100
      + w_degree_variance × degree_variance
      + w_hub_concentration × hub_concentration
      + w_clustering_coeff × clustering_coeff
      + w_avg_path_length × avg_path_length / 10
      + w_diameter × diameter_estimate / 50
      + w_community_count × log₁₀(community_count + 1)
      + w_packing_factor × packing_factor
      + w_forward_edge_fraction × forward_edge_fraction
      + w_working_set_ratio × log₂(working_set_ratio + 1)
      + w_reorder_time × reorder_time
      + w_dv_x_hub × (degree_variance × hub_concentration)   # quadratic
      + w_mod_x_logn × (modularity × log_nodes)              # quadratic
      + w_pf_x_wsr × (packing_factor × log₂(wsr + 1))       # quadratic
      + cache_l1_impact × 0.5 + cache_l2_impact × 0.3        # cache
      + cache_l3_impact × 0.2 + cache_dram_penalty           # cache
      + (w_fef_convergence × fef if PR/SSSP)                  # convergence
```

The algorithm with the highest score is selected. Actual weight values are stored in `results/weights/type_N/weights.json` — run `--train` to generate them from your benchmark data.

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
| 15 | LeidenOrder | Basic Leiden ordering via GVE-Leiden (baseline reference) |

> **Note:** For current variant lists, see `scripts/lib/utils.py`: `RABBITORDER_VARIANTS`, `GORDER_VARIANTS`, `RCM_VARIANTS`, `GRAPHBREW_VARIANTS`. Use `get_algo_variants(algo_id)` to query programmatically.

---

## Tuning Strategies

### Strategy 1: Favor One Algorithm

Make GraphBrewOrder almost always win:

```json
{
  "GraphBrewOrder": {
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

ORIGINAL with high bias + negative `w_log_nodes`/`w_log_edges` wins for small communities; GraphBrewOrder with positive log weights wins for large ones.

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
nano results/weights/type_0/weights.json
python3 -c "import json; json.load(open('results/weights/type_0/weights.json'))"
```

---

## Validating Weights

```bash
# Check what AdaptiveOrder selects
./bench/bin/pr -f graph.el -s -o 14 -n 1 2>&1 | grep -A 20 "Adaptive Reordering"
```

Output shows algorithm selection details (features + selected algorithm).

```bash
# Verbose score details
./bench/bin/pr -f graph.el -s -o 14 -n 1 -v
```

---

## Default Weights

If no weights file exists, hardcoded defaults favour ORIGINAL for small communities (high bias, negative log-size weights) and GraphBrewOrder for large, modular, hub-heavy communities (positive `w_modularity`, `w_hub_concentration`, log-size weights).

The **bias** is the most important weight — computed as `0.5 × avg_speedup_vs_RANDOM`. Higher bias means the algorithm is generally faster.

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

The 8-phase `--train` mode fills: (0) graph analysis, (1) reorder_time, (2) bias/scale, (3) cache impacts, (4) base weights, (5) topology weights, (6) benchmark multipliers, (7) per-graph-type files.

See [[AdaptiveOrder-ML#training-the-perceptron]] for iterative training and progression recommendations.

---

## Gradient Update Rule

Online SGD with error signal `error = (speedup - 1.0) - current_score`:

```python
lr = 0.1   # default learning rate
for feature in all_features:  # linear + quadratic + convergence
    weights[feature] += lr * error * feature_value
for key in w_and_cache_keys:
    weights[key] *= (1.0 - 1e-4)  # L2 regularization
```

Positive error → increase weights for features that predicted success. Negative error → decrease. ORIGINAL is trained like any other algorithm, allowing the model to learn when *not reordering* is optimal.

> **Note:** Modularity now uses the real value from graph features (computed via Leiden or loaded from cache), with `clustering_coefficient × 1.5` as a fallback if modularity is unavailable.

---

## Multi-Restart Training & Benchmark Multipliers

**Multi-restart perceptrons** (5 restarts × 800 epochs per benchmark) avoid local minima by averaging weights across independent runs with different seeds. Features are z-score normalized for stable gradients.

**Regret-aware grid search** optimizes per-benchmark multipliers (`benchmark_weights`) after base weight training. For each algorithm, it tests 30 random multiplier candidates from a log-spaced grid [0.1, 10.0], keeping combinations that maximize accuracy while minimizing regret (performance loss vs optimal).

**Variant pre-collapse:** Before saving, only the highest-bias variant per base algorithm is kept (e.g., `GraphBrewOrder_leiden_dfs` beats `GraphBrewOrder_leiden` → saved as `GraphBrewOrder`).

---

## Validation

Run `--eval-weights` to measure current accuracy, median regret, top-2 accuracy, and unique predictions on your data.

See [[Python-Scripts#-eval_weightspy---weight-evaluation--c-scoring-simulation]] for the `eval_weights.py` tool.

### Cross-Validation (LOGO)

Leave-One-Graph-Out CV trains on all-but-one graph, predicts the held-out graph, and repeats. An `overfitting_score > 0.2` suggests too-specialized weights — add more graphs or increase L2 regularization.

```python
from scripts.lib.weights import cross_validate_logo
result = cross_validate_logo(benchmark_results, reorder_results=reorder_results, weights_dir=weights_dir)
print(f"LOGO: {result['accuracy']:.1%}, Overfit: {result['overfitting_score']:.2f}")
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Algorithm X never selected | Check bias: `grep -A1 '"AlgoX"' results/weights/type_0/weights.json` — increase bias or feature weights |
| Wrong algorithm selected | Run with `-v` to see scores, manually verify feature × weight products |
| Weights not loading | Check `ls results/weights/type_*/weights.json` and `echo $PERCEPTRON_WEIGHTS_FILE` |

**Best practices:** Start with moderate biases (0.5–0.7), validate empirically after tuning, keep backups (`cp -r results/weights/ results/weights.backup/`).

---

## Next Steps

- [[Correlation-Analysis]] - Automatic weight generation
- [[AdaptiveOrder-ML]] - Using the perceptron model
- [[Adding-New-Algorithms]] - Add algorithms to the perceptron

---

[← Back to Home](Home) | [GraphBrewOrder →](GraphBrewOrder)
