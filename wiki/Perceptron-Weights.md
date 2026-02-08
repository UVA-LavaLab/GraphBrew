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

```
INPUTS (Features)              WEIGHTS                    OUTPUT
=================              =======                    ======

--- Linear Features ---
modularity: 0.72       --*---> w_mod: 0.28 ----------+
density: 0.001         --*---> w_den: -0.15 ---------+
degree_var: 2.1        --*---> w_dv: 0.18 -----------+
hub_conc: 0.45         --*---> w_hc: 0.22 -----------+
cluster_coef: 0.3      --*---> w_cc: 0.12 -----------+
avg_path: 5.2          --*---> w_ap: 0.08 -----------+
diameter: 16           --*---> w_di: 0.05 -----------+
packing_factor: 0.6    --*---> w_pf: 0.10 -----------+----> SUM
fwd_edge_frac: 0.4     --*---> w_fef: 0.08 ----------+    (+bias)
working_set_ratio: 3.2 --*---> w_wsr: 0.12 ----------+      |
                                                      |      |
--- Quadratic Cross-Terms ---                         |      |
dv × hub: 0.95         --*---> w_dv_x_hub: 0.15 -----+      |
mod × logN: 2.88       --*---> w_mod_x_logn: 0.06 ---+      +---> SCORE
pf × log₂(wsr+1): 1.27 --*--> w_pf_x_wsr: 0.09 -----+      |
                                                      |      |
--- Convergence Bonus (PR/SSSP only) ---              |      |
fwd_edge_frac: 0.4     --*---> w_fef_conv: 0.05 -----+------+


ALGORITHM SELECTION:
====================
RABBITORDER:    score = 2.31  <-- WINNER
LeidenCSR:      score = 2.18
HubClusterDBG:  score = 1.95
GORDER:         score = 1.82
ORIGINAL:       score = 0.50

SAFETY CHECKS:
==============
1. OOD Guardrail: If type distance > 1.5 → force ORIGINAL
2. ORIGINAL Margin: If best_score - ORIGINAL_score < 0.05 → keep ORIGINAL
```

**Key Insight:** Each weight encodes "When this feature is high, how well does this algorithm perform?"

---

## Weight File Location

### Default Location
```
GraphBrew/scripts/weights/
├── active/               # C++ reads from here
│   ├── type_registry.json # Maps graphs → types + centroids
│   ├── type_0.json        # Cluster 0 weights
│   ├── type_1.json        # Cluster 1 weights
│   └── type_N.json        # Additional clusters
├── merged/               # Accumulated weights from all runs
└── runs/                 # Historical run snapshots
    └── run_YYYYMMDD_*/
```

### Automatic Clustering and Storage

When weights are saved, the script automatically:
1. Creates type-based weight files in `scripts/weights/active/` (e.g., `type_0.json`, `type_1.json`)
2. Updates the **type registry** (`scripts/weights/active/type_registry.json`) with cluster centroids
3. Optionally saves a snapshot to `scripts/weights/runs/` for merging later

This ensures weights are never accidentally overwritten and the best cluster is selected at runtime.

### Auto-Clustering Type System

AdaptiveOrder uses an automatic clustering system that groups graphs by feature similarity instead of predefined categories. This allows the system to scale to any number of graph types.

**How It Works:**
1. **Feature Extraction:** For each graph, compute 7 clustering features: modularity, log_nodes, log_edges, avg_degree, degree_variance, hub_concentration, clustering_coefficient
2. **Clustering:** Group similar graphs using k-means-like clustering
3. **Per-Cluster Training:** Train optimized weights for each cluster
4. **Runtime Matching:** Select best cluster based on Euclidean distance to centroid

**Type Files:**
```
scripts/weights/
├── active/               # C++ reads from here
│   ├── type_registry.json
│   └── type_N.json
├── merged/               # Merged weights from all runs
└── runs/                 # Historical snapshots
```

**Loading Order:**
1. Environment variable `PERCEPTRON_WEIGHTS_FILE` (if set)
2. Best matching type file from `scripts/weights/active/type_N.json`
3. Semantic type fallback (if type files don't exist)
4. Hardcoded defaults

**Example Output:**
```
Best matching type: type_0 (distance: 0.12)
Perceptron: Loaded 5 weights from scripts/weights/active/type_0.json
```

### Environment Override
```bash
export PERCEPTRON_WEIGHTS_FILE=/path/to/custom_weights.json
./bench/bin/pr -f graph.el -s -o 14 -n 3
```

### Fallback Behavior
If no type files exist, C++ uses hardcoded defaults with conservative weights that favor ORIGINAL for small communities and LeidenCSR for larger ones.

---

## File Structure

### Complete Example (Enhanced Format)

```json
{
  "ORIGINAL": {
    "bias": 0.5,
    "w_modularity": 0.1,
    "w_density": 0.05,
    "w_degree_variance": 0.03,
    "w_hub_concentration": 0.05,
    "w_log_nodes": 0.02,
    "w_log_edges": 0.02,
    "w_clustering_coeff": 0.04,
    "w_avg_path_length": 0.02,
    "w_diameter": 0.01,
    "w_community_count": 0.03,
    "w_packing_factor": 0.0,
    "w_forward_edge_fraction": 0.0,
    "w_working_set_ratio": 0.0,
    "w_dv_x_hub": 0.0,
    "w_mod_x_logn": 0.0,
    "w_pf_x_wsr": 0.0,
    "w_fef_convergence": 0.0,
    "w_reorder_time": 0.0,
    "cache_l1_impact": 0,
    "cache_l2_impact": 0,
    "cache_l3_impact": 0,
    "cache_dram_penalty": 0,
    "benchmark_weights": {
      "pr": 1.0,
      "bfs": 1.0,
      "cc": 1.0,
      "sssp": 1.0,
      "bc": 1.0,
      "tc": 1.0
    },
    "_metadata": {
      "win_rate": 0.0,
      "avg_speedup": 1.0,
      "sample_count": 0,
      "avg_reorder_time": 0.0
    }
  },
  "LeidenCSR": {
    "bias": 3.5,
    "w_modularity": 0.1,
    "w_density": 0.05,
    "w_degree_variance": 0.03,
    "w_hub_concentration": 0.05,
    "w_log_nodes": 0.02,
    "w_log_edges": 0.02,
    "w_clustering_coeff": 0.04,
    "w_avg_path_length": 0.02,
    "w_diameter": 0.01,
    "w_community_count": 0.03,
    "w_packing_factor": 0.08,
    "w_forward_edge_fraction": 0.05,
    "w_working_set_ratio": 0.12,
    "w_dv_x_hub": 0.15,
    "w_mod_x_logn": 0.06,
    "w_pf_x_wsr": 0.09,
    "w_fef_convergence": 0.04,
    "w_reorder_time": -0.0087,
    "cache_l1_impact": 0.00021,
    "cache_l2_impact": 0.00019,
    "cache_l3_impact": 0.00021,
    "cache_dram_penalty": -7.9e-05,
    "benchmark_weights": {
      "pr": 1.0,
      "bfs": 1.0,
      "cc": 1.0,
      "sssp": 1.0,
      "bc": 1.0,
      "tc": 1.0
    },
    "_metadata": {
      "win_rate": 0.8,
      "avg_speedup": 1.45,
      "sample_count": 20,
      "avg_reorder_time": 8.7
    }
  },
  "_metadata": {
    "enhanced_features": true,
    "last_updated": "2026-02-02T12:00:00"
  }
}
```

Note: Algorithm names in the weights file match the names from `scripts/lib/utils.py` ALGORITHMS dict (e.g., `ORIGINAL`, `LeidenCSR`, `RABBITORDER`).

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
base_score = bias
           + Σ(w_feature × feature_value)           # linear features
           + w_dv_x_hub × dv × hub                   # quadratic: degree_var × hub_conc
           + w_mod_x_logn × mod × logN               # quadratic: modularity × log_nodes
           + w_pf_x_wsr × pf × log₂(wsr+1)           # quadratic: packing × cache pressure

# Convergence bonus (PR/SSSP only)
if benchmark ∈ {PR, SSSP}:
    base_score += w_fef_convergence × forward_edge_fraction

final_score = base_score × benchmark_weights[benchmark_type]
```

For `BENCH_GENERIC` (default when no benchmark is specified), the multiplier is 1.0.

### Safety Checks

**OOD (Out-of-Distribution) Guardrail:**
If the graph's Euclidean distance to the nearest type centroid exceeds `1.5`, the system returns `ORIGINAL` instead of risking a bad prediction on an unfamiliar graph type. The 7-dimensional feature space has a max distance of √7 ≈ 2.65, so 1.5 is a meaningful threshold.

**ORIGINAL Margin Fallback:**
If the best-scoring algorithm's score exceeds ORIGINAL's score by less than `ORIGINAL_MARGIN_THRESHOLD = 0.05`, the system keeps `ORIGINAL` to avoid reordering overhead for marginal gains.

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

Use simpler algorithms for small communities:

```json
{
  "ORIGINAL": {
    "bias": 0.8,
    "w_log_nodes": -0.3,
    "w_log_edges": -0.3
  },
  "LeidenCSR": {
    "bias": 0.5,
    "w_log_nodes": 0.2,
    "w_log_edges": 0.2
  }
}
```

Result:
- Small communities (< 100 nodes): ORIGINAL wins
- Large communities (> 1000 nodes): LeidenCSR wins

### Strategy 3: Structure-Based Selection

Use hub algorithms for hub-heavy graphs:

```json
{
  "HUBCLUSTERDBG": {
    "bias": 0.6,
    "w_hub_concentration": 0.5,
    "w_degree_variance": 0.3
  },
  "RCM": {
    "bias": 0.6,
    "w_hub_concentration": -0.4,
    "w_degree_variance": -0.3
  }
}
```

Result:
- High hub concentration: HUBCLUSTERDBG wins
- Low hub concentration: RCM wins

### Strategy 4: Workload-Specific

For PageRank (benefits from hub locality):

```json
{
  "LeidenCSR": {
    "bias": 0.85,
    "w_hub_concentration": 0.35
  }
}
```

For BFS (benefits from bandwidth reduction):

```json
{
  "RCM": {
    "bias": 0.75,
    "w_density": 0.2
  }
}
```

---

## Generating Weights

### Automatic: Using graphbrew_experiment.py

```bash
# Full pipeline with weight generation
python3 scripts/graphbrew_experiment.py --full --size small

# Comprehensive weight training
python3 scripts/graphbrew_experiment.py --train --size medium --auto
```

### Manual: Edit JSON Directly

```bash
# Edit the file
nano scripts/weights/active/type_0.json

# Validate JSON
python3 -c "import json; json.load(open('scripts/weights/active/type_0.json'))"
```

### Hybrid: Start from Auto-Generated, Then Tune

```bash
# Generate weights from benchmark data
python3 scripts/graphbrew_experiment.py --phase weights

# Backup
cp scripts/weights/active/type_0.json scripts/weights/active/type_0.json.backup

# Edit manually to adjust biases
vim scripts/weights/active/type_0.json
```

---

## Validating Weights

### Check Current Selections

```bash
./bench/bin/pr -f graph.el -s -o 14 -n 1 2>&1 | grep -A 20 "Adaptive Reordering"
```

Output shows what was selected:
```
=== Adaptive Reordering Selection (Depth 0, Modularity: 0.835) ===
Comm    Nodes   Edges   Density DegVar  HubConc Selected
0       3500    45000   0.0073  89.3    0.62    LeidenCSR
1       2800    28000   0.0071  34.2    0.41    LeidenCSR
2       600     3200    0.0178  2.1     0.09    ORIGINAL
```

### Debug Score Calculation

Add verbose output:
```bash
./bench/bin/pr -f graph.el -s -o 14 -n 1 -v
```

### Unit Test Weights

```python
#!/usr/bin/env python3
import json

def test_weights(weights_file):
    with open(weights_file) as f:
        weights = json.load(f)
    
    # Test case: large modular community
    features = {
        "modularity": 0.8,
        "log_nodes": 4.0,
        "log_edges": 5.0,
        "density": 0.01,
        "avg_degree": 0.25,
        "degree_variance": 0.5,
        "hub_concentration": 0.4
    }
    
    scores = {}
    for algo, w in weights.items():
        score = w.get("bias", 0.5)
        for feat, val in features.items():
            score += w.get(f"w_{feat}", 0) * val
        scores[algo] = score
    
    best = max(scores.items(), key=lambda x: x[1])
    print(f"Best algorithm: {best[0]} (score: {best[1]:.3f})")
    
    for algo, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {algo}: {score:.3f}")

test_weights("scripts/weights/active/type_0.json")
```

---

## Default Weights

If no weights file exists, these defaults are used:

```json
{
  "ORIGINAL": {
    "bias": 0.6,
    "w_modularity": 0.0,
    "w_log_nodes": -0.1,
    "w_log_edges": -0.1,
    "w_density": 0.1,
    "w_avg_degree": 0.0,
    "w_degree_variance": 0.0,
    "w_hub_concentration": 0.0
  },
  "LeidenCSR": {
    "bias": 0.85,
    "w_modularity": 0.25,
    "w_log_nodes": 0.1,
    "w_log_edges": 0.1,
    "w_density": -0.05,
    "w_avg_degree": 0.15,
    "w_degree_variance": 0.15,
    "w_hub_concentration": 0.25
  }
}
```

These defaults:
- Favor ORIGINAL for small communities
- Favor LeidenCSR for large, modular, hub-heavy communities

---

## Understanding Algorithm Biases

The **bias** is the most important weight - it's the base preference for an algorithm before any graph features are considered. Higher bias = more likely to be selected.

### How Bias is Calculated

When training weights, the bias is computed from benchmark performance:

```
bias = 0.5 × avg_speedup_vs_RANDOM
```

Where:
- **RANDOM (ID 1)** is the baseline - represents worst-case random node ordering
- `avg_speedup` = RANDOM_time / Algorithm_time

### Example Bias Values

After training on 87 graphs, typical biases look like:

| Algorithm | Bias | Interpretation |
|-----------|------|----------------|
| HUBSORT | ~26.0 | **26x faster** than random on average |
| SORT | ~9.5 | 9.5x faster than random |
| HUBSORTDBG | ~8.5 | 8.5x faster than random |
| LeidenDendrogram | ~2.4 | 2.4x faster than random |
| LeidenCSR | ~2.3 | 2.3x faster than random |
| LeidenOrder | ~1.8 | 1.8x faster than random |
| ORIGINAL | 0.5 | Baseline (no reordering) |
| RANDOM | 0.5 | Baseline (1.00x) |
| DBG | ~0.44 | Slightly worse than random |
| RABBITORDER | ~0.40 | Slower than random on small graphs |

### Why Some Algorithms Have Low Bias

Algorithms like `RABBITORDER`, `DBG`, and `LeidenCSR` may show biases < 0.5, meaning they're slower than RANDOM on the training graphs. This happens because:

1. **Reordering overhead** - Complex algorithms have high setup cost
2. **Small graph penalty** - Sophisticated ordering doesn't help small graphs
3. **Graph type mismatch** - Algorithm designed for different graph structures

### Adjusting Biases Manually

To favor an algorithm regardless of graph features:

```json
{
  "LeidenCSR": {
    "bias": 3.0,  // Force higher preference
    "w_modularity": 0.1,
    ...
  }
}
```

---

## Step-by-Step Terminal Training Guide

### Training Pipeline Diagram

```
+=====================================================+
|            TRAINING PIPELINE (Python)               |
+=====================================================+

+-----------+    +-----------+    +-----------+
|  Graph 1  |    |  Graph 2  |    |  Graph N  |
+-----+-----+    +-----+-----+    +-----+-----+
      |               |               |
      +-------+-------+-------+-------+
              |
              v
      +-------+-------+
      | PHASE 1:      |
      | Reorder each  |
      | graph with    |
      | all 18 algos  |
      | (58 variants) |
      +-------+-------+
              |
              v
      +-------+-------+
      | PHASE 2:      |
      | Run benchmarks|
      | PR,BFS,CC,    |
      | SSSP,BC       |
      +-------+-------+
              |
              v
      +-------+--------+
      | PHASE 3:        |
      | compute_weights |
      | _from_results() |
      +-------+---------+
              |
     +--------+--------+
     |                  |
     v                  v
+----+-----+    +------+------+
| Multi-   |    | Regret-Aware|
| Restart  |    | Grid Search |
| Percep-  |    | (benchmark  |
| trons    |    |  multipliers)|
| (5×800   |    | 30 iters ×  |
|  epochs  |    | 32 values)  |
|  per     |    +------+------+
|  bench)  |           |
+----+-----+           |
     |                  |
     v                  v
+----+-----------------+----+
| Pre-collapse variants:    |
| keep highest-bias variant |
| per base algorithm        |
+----+----------------------+
     |
     v
+----+------+---+
| type_0.json   |
| (C++ runtime) |
+---------------+
```

### Complete Training Workflow

#### Step 1: Clean Previous Results (Optional)

```bash
cd /path/to/GraphBrew

# Backup current weights
cp -r scripts/weights/ scripts/weights.backup/

# Clean results but keep graphs
python3 scripts/graphbrew_experiment.py --clean

# Or clean all weights
rm -rf scripts/weights/active/* scripts/weights/merged/* scripts/weights/runs/*
```

#### Step 2: Generate Reorderings for All Graphs

```bash
# Small graphs only (~5 minutes)
python3 scripts/graphbrew_experiment.py \
    --graphs-dir ./results/graphs \
    --size small \
    --phase reorder \
    --generate-maps

# All graphs including large (~1-2 hours)
python3 scripts/graphbrew_experiment.py \
    --graphs-dir ./results/graphs \
    --max-graphs 50 \
    --phase reorder \
    --generate-maps
```

#### Step 3: Run Benchmarks (RANDOM Baseline)

```bash
# All benchmarks on all graphs
python3 scripts/graphbrew_experiment.py \
    --graphs-dir ./results/graphs \
    --max-graphs 50 \
    --phase benchmark \
    --benchmarks pr bfs cc sssp bc \
    --trials 2 \
    --use-maps
```

#### Step 4: Generate New Weights

```bash
# Generate weights from benchmark results
python3 scripts/graphbrew_experiment.py \
    --graphs-dir ./results/graphs \
    --phase weights
```

#### Step 5: Run Brute-Force Validation

```bash
# Test all 18 algorithms vs adaptive choice
python3 scripts/graphbrew_experiment.py \
    --max-graphs 50 \
    --brute-force \
    --validation-benchmark pr \
    --trials 2
```

#### Step 6: View Results

```bash
# Check type weight files
ls -la scripts/weights/active/

# Check algorithm biases in type_0.json
cat scripts/weights/active/type_0.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Algorithm Biases (sorted by speedup):')
items = [(k, v.get('bias', 0)) for k, v in d.items() if not k.startswith('_')]
for k, b in sorted(items, key=lambda x: -x[1])[:15]:
    print(f'  {k:20s}: {b:.3f}')
"

# Check validation accuracy
cat results/brute_force_*.json | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f'Accuracy: {d.get(\"overall_accuracy\", \"N/A\")}')
" 2>/dev/null || echo "Run brute-force first"
```

### Quick Training (Minimal)

For fast iteration during development:

```bash
# Quick training on 3 small graphs, 1 trial each
python3 scripts/graphbrew_experiment.py \
    --size small \
    --max-graphs 3 \
    --trials 1 \
    --skip-cache \
    2>&1 | tee /tmp/training.log
```

### Complete Training Pipeline

If many weight fields show as 0 or default 1.0, use `--train` to run the complete pipeline:

```bash
# Complete training: reorder → benchmark → cache sim → update weights
python3 scripts/graphbrew_experiment.py \
    --train \
    --size small \
    --max-graphs 5 \
    --trials 2
```

This runs all 6 phases sequentially:
1. **Phase 1 (Reorderings)**: Fills `w_reorder_time`
2. **Phase 2 (Benchmarks)**: Fills `bias`, `w_log_edges`, `w_avg_degree`  
3. **Phase 3 (Cache Sim)**: Fills `cache_l1_impact`, `cache_l2_impact`, `cache_l3_impact`
4. **Phase 4 (Base Weights)**: Fills `w_density`, `w_degree_variance`, `w_hub_concentration`
5. **Phase 5 (Topology)**: Fills `w_clustering_coeff`, `w_avg_path_length`, `w_diameter`
6. **Phase 6 (Benchmark Weights)**: Fills `benchmark_weights.{pr,bfs,cc,sssp,bc,tc}`

---

## Gradient Update Rule: Online Learning

### Error Signal Computation

```python
error = (speedup - 1.0) - current_score
# Positive error: algorithm performed better than expected
# Negative error: algorithm performed worse than expected
```

### Weight Update (Stochastic Gradient Descent)

```python
learning_rate = 0.01

# Core feature weights
w_modularity      += lr * error * modularity
w_density         += lr * error * density  
w_degree_variance += lr * error * degree_variance
w_hub_concentration += lr * error * hub_concentration

# New graph-aware features
w_packing_factor         += lr * error * packing_factor
w_forward_edge_fraction  += lr * error * forward_edge_fraction
w_working_set_ratio      += lr * error * log2(working_set_ratio + 1)

# Quadratic interaction weights
w_dv_x_hub   += lr * error * (degree_variance * hub_concentration)
w_mod_x_logn += lr * error * (modularity * log_nodes)
w_pf_x_wsr   += lr * error * (packing_factor * log2(working_set_ratio + 1))

# Convergence weight (PR/SSSP benchmarks only)
if benchmark in ('pr', 'sssp'):
    w_fef_convergence += lr * error * forward_edge_fraction

# Extended topology weights
w_avg_path_length += lr * error * (avg_path_length / 10)
w_diameter        += lr * error * (diameter / 50)
w_clustering_coeff += lr * error * clustering_coeff

# Cache impact weights (when simulation data available)
w_cache_l1_impact += lr * error * l1_hit_rate
w_cache_dram_penalty += lr * error * dram_penalty

# L2 Regularization (weight decay)
WEIGHT_DECAY = 1e-4
for key in all_w_and_cache_keys:
    weights[key] *= (1.0 - WEIGHT_DECAY)
```

> **L2 Regularization:** After each gradient update, all `w_*` and `cache_*` weights are decayed by a factor of `(1.0 - 1e-4)`. This prevents weight explosion during long training runs and improves generalization.

### When Algorithm Performs Well (speedup > 1.0)

```python
error = speedup - 1.0 - current_score  # positive

# Increase weights for features that predicted this success
w_modularity += learning_rate * error * modularity
w_hub_concentration += learning_rate * error * hub_concentration
# ... etc for all features (including new ones)
```

**Effect**: Algorithm gets higher scores for similar graphs in future.

### When Algorithm Performs Poorly (speedup < 1.0)

```python
error = speedup - 1.0 - current_score  # negative

# Decrease weights (error is negative)
w_modularity += learning_rate * error * modularity
```

**Effect**: Algorithm gets lower scores for similar graphs in future.

### ORIGINAL Is Now Trained

Previously, `ORIGINAL` was skipped during correlation-based weight training. It is now trained like any other algorithm, allowing the perceptron to learn when *not reordering* is the best choice (e.g., small graphs, already well-ordered graphs, or graphs with weak community structure).

---

## Multi-Restart Perceptron Training

The `compute_weights_from_results()` function uses **multi-restart perceptrons** to avoid local minima and produce stable weights.

### Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `N_RESTARTS` | 5 | Independent training runs per benchmark |
| `N_EPOCHS` | 800 | Gradient steps per restart |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization factor |
| `LEARNING_RATE` | 0.01 | SGD step size |

### How It Works

```python
for bench in ['pr', 'bfs', 'cc', 'sssp']:
    all_weights = []
    for restart in range(N_RESTARTS):
        seed = 42 + restart * 1000 + bench_index * 100  # Deterministic
        rng = random.Random(seed)
        
        # Initialize weights near zero
        weights = {algo: small_random_values(rng) for algo in algorithms}
        
        for epoch in range(N_EPOCHS):
            # Z-score normalize features for stable gradients
            features = z_score_normalize(raw_features)
            
            # SGD: update weights based on speedup error
            for graph in training_graphs:
                error = (speedup - 1.0) - current_score
                weights[algo] += lr * error * features
                
            # L2 regularization
            weights *= (1.0 - WEIGHT_DECAY)
        
        all_weights.append(weights)
    
    # Average across all restarts
    final_weights[bench] = average(all_weights)

# Average across benchmarks for scoreBase()
scoreBase_weights = average(final_weights['pr'], final_weights['bfs'], ...)
```

### Why Multi-Restart?

Single-run perceptrons are sensitive to:
- **Weight initialization**: Different starting points find different optima
- **Training order**: SGD shuffling creates variance
- **Feature scaling**: Z-score normalization helps but doesn't eliminate all sensitivity

By averaging 5 restarts, the bias and feature weights stabilize, reducing variance by ~√5.

---

## Regret-Aware Grid Search (Benchmark Multipliers)

After training `scoreBase()` weights, the pipeline optimizes per-benchmark multipliers (`benchmark_weights`) using a **regret-aware grid search**.

### What Are Benchmark Multipliers?

The C++ score for algorithm `A` on benchmark `B` is:
```
finalScore(A, B) = scoreBase(A, features) × benchmarkMultiplier(A, B)
```

Different algorithms may excel at different benchmarks. The multipliers capture this:
- `benchmarkMultiplier(LeidenCSR, pr) = 1.2` → boost for PageRank
- `benchmarkMultiplier(DBG, bfs) = 0.8` → penalty for BFS

### Grid Search Process

```python
# For each algorithm:
for algo in algorithms:
    best_multipliers = {b: 1.0 for b in benchmarks}
    best_objective = (-inf, inf)  # (accuracy, mean_regret)
    
    for iteration in range(30):
        # Random multiplier candidate from 32 log-spaced values [0.1, 10.0]
        candidate = random.choice(MULTIPLIER_GRID)
        target_bench = random.choice(benchmarks)
        
        trial = best_multipliers.copy()
        trial[target_bench] = candidate
        
        # Simulate C++ scoring with these multipliers
        accuracy, mean_regret = simulate_all_predictions(
            scoreBase_weights, trial, graph_features, benchmark_results
        )
        
        # Keep if better: higher accuracy OR same accuracy with lower regret
        if (accuracy, -mean_regret) > best_objective:
            best_multipliers = trial
            best_objective = (accuracy, -mean_regret)
    
    weights[algo]['benchmark_weights'] = best_multipliers
```

### Why Regret-Aware?

Pure accuracy optimization can lead to degenerate solutions (e.g., always predicting the same algorithm). By jointly optimizing `(accuracy, −mean_regret)`, the search:
- **Prefers diverse predictions** that cover different graph types
- **Penalizes catastrophic mispredictions** even when accuracy is high
- **Produces practical weights** that minimize real-world performance loss

### Variant Pre-Collapse

Before saving to `type_0.json`, algorithm variants are collapsed:
- Only the **highest-bias variant** per base algorithm is kept
- Example: `LeidenCSR_gve` (bias=0.8), `LeidenCSR_gveopt2` (bias=0.9) → only `LeidenCSR` (bias=0.9) saved
- The C++ `ParseWeightsFromJSON()` similarly keeps the highest-bias variant when loading

---

## Validating Weights with eval_weights.py

After training, validate weights by simulating C++ scoring:

```bash
python3 scripts/eval_weights.py
```

This reports:
- **Accuracy**: % of correct predictions (base-algorithm level)
- **Regret**: How much slower the predicted algorithm is vs the actual best
- **Top-2 accuracy**: % where prediction is in top 2
- **Per-benchmark breakdown**: Accuracy per benchmark type

Current metrics (47 graphs × 4 benchmarks = 188 predictions):
- **46.8% accuracy** (88/188 correct)
- **2.6% base-aware median regret** (selected algorithm is typically within 2.6% of optimal)
- **64.9% top-2 accuracy**
- **13 unique predictions** across all graphs/benchmarks

See [[Python-Scripts#-eval_weightspy---weight-evaluation--c-scoring-simulation]] for details.

---

### Full Training (Production)

For production-quality weights:

```bash
# Full training on all graphs with validation
python3 scripts/graphbrew_experiment.py \
    --graphs-dir ./results/graphs \
    --max-graphs 50 \
    --benchmarks pr bfs cc sssp bc \
    --trials 3 \
    --generate-maps \
    --use-maps \
    2>&1 | tee results/training_full.log

# Follow with brute-force validation
python3 scripts/graphbrew_experiment.py \
    --max-graphs 50 \
    --brute-force \
    --validation-benchmark pr \
    --trials 2 \
    2>&1 | tee results/validation.log
```

### Monitoring Long Runs

```bash
# In another terminal, monitor progress
tail -f results/training_full.log

# Check if still running
ps aux | grep graphbrew

# Kill if needed
pkill -f graphbrew_experiment
```

---

## Cross-Validation (LOGO)

To assess whether trained weights generalize well to unseen graphs, use **Leave-One-Graph-Out (LOGO) cross-validation**:

```python
from scripts.lib.weights import cross_validate_logo

result = cross_validate_logo(
    benchmark_results,     # benchmark data from experiments
    graph_features,        # per-graph feature dicts
    type_registry           # type registry with centroids
)

print(f"LOGO Accuracy:     {result['accuracy']:.1%}")
print(f"Full-Train Accuracy: {result['full_training_accuracy']:.1%}")
print(f"Overfitting Score: {result['overfitting_score']:.2f}")
```

**How it works:**
1. For each graph, train weights on all *other* graphs
2. Predict the best algorithm for the held-out graph
3. Compare prediction to actual best algorithm
4. Repeat for every graph → compute accuracy

**Output fields:**
| Field | Description |
|-------|-------------|
| `accuracy` | % of held-out graphs where prediction was correct |
| `full_training_accuracy` | % when trained on all graphs (upper bound) |
| `overfitting_score` | `full_training_accuracy - accuracy` (higher = more overfitting) |
| `per_graph` | Detailed predictions for each held-out graph |

**Warning:** An `overfitting_score > 0.2` suggests the weights are too specialized to the training set. Consider:
- Adding more diverse graphs to the training set
- Increasing L2 regularization (`WEIGHT_DECAY`)
- Reducing the number of features or interactions

---

## Troubleshooting

### "Algorithm X never gets selected"

Check if bias is too low:
```bash
cat scripts/weights/active/type_0.json | grep -A 1 '"AlgorithmX"'
```

Increase bias or relevant weights.

### "Wrong algorithm selected"

1. Check feature values:
```bash
./bench/bin/pr -f graph.el -s -o 14 -n 1 2>&1 | grep "Comm"
```

2. Manually calculate scores with those features

3. Adjust weights accordingly

### "Weights file not loading"

Check file exists and is valid JSON:
```bash
ls -la scripts/weights/active/type_*.json
python3 -c "import json; json.load(open('scripts/weights/active/type_0.json'))"
```

Check environment variable:
```bash
echo $PERCEPTRON_WEIGHTS_FILE
```

---

## Best Practices

### 1. Start Conservative

Begin with moderate biases (0.5-0.7) and small weights (±0.1-0.2).

### 2. Validate Empirically

Always benchmark after tuning:
```bash
# Before tuning
./bench/bin/pr -f graph.el -s -o 14 -n 10

# After tuning
./bench/bin/pr -f graph.el -s -o 14 -n 10
```

### 3. Keep Backups

```bash
cp -r scripts/weights/ scripts/weights.backup/
```

### 4. Document Changes

Add comments (JSON doesn't support comments, but you can use a README):
```bash
echo "2026-01-18: Increased LeidenCSR bias for PageRank workload" >> scripts/weights_changelog.txt
```

---

## Next Steps

- [[Correlation-Analysis]] - Automatic weight generation
- [[AdaptiveOrder-ML]] - Using the perceptron model
- [[Adding-New-Algorithms]] - Add algorithms to the perceptron

---

[← Back to Home](Home) | [GraphBrewOrder →](GraphBrewOrder)
