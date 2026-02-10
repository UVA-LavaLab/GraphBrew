# AdaptiveOrder: ML-Powered Algorithm Selection

AdaptiveOrder (algorithm 14) uses a **machine learning perceptron** to automatically select the best reordering algorithm for each community in your graph. This page explains how it works and how to train it.

## Overview

Instead of using one reordering algorithm for the entire graph, AdaptiveOrder:
1. **Computes graph features** (modularity, degree variance, hub concentration, etc.)
2. **Finds best matching type** from auto-clustered type files using Euclidean distance
3. **Loads specialized weights** for that type (type_0.json, type_1.json, etc.)
4. **Detects communities** using Leiden
5. **Computes features** for each community
6. **Uses a trained perceptron** to predict the best algorithm
7. **Applies different algorithms** to different communities

## Command-Line Format

```bash
# Format: -o 14[:max_depth[:resolution[:min_recurse_size[:mode]]]]

# Default: per-community selection, no recursion
./bench/bin/pr -f graph.el -s -o 14 -n 3

# Multi-level adaptive: recurse up to depth 2
./bench/bin/pr -f graph.el -s -o 14:2 -n 3

# With custom resolution (more communities)
./bench/bin/pr -f graph.el -s -o 14:1:1.0 -n 3

# Full-graph mode: pick single best algorithm for entire graph
./bench/bin/pr -f graph.el -s -o 14:0:0.75:50000:1 -n 3
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 0 | Max recursion depth (0 = no recursion, 1+ = multi-level) |
| `resolution` | auto | Leiden resolution (auto: continuous formula with CV guardrail) |
| `min_recurse_size` | 50000 | Minimum community size for recursion |
| `mode` | 0 | 0 = per-community, 1 = full-graph adaptive |

**Auto-Resolution Formula:**
```
γ = clip(0.5 + 0.25 × log₁₀(avg_degree + 1), 0.5, 1.2)
If CV(degree) > 2: γ = max(γ, 1.0)  // CV guardrail for power-law graphs
```
*Heuristic for stable partitions; users should sweep γ for best community quality.*

## Operating Modes

### Mode 0: Per-Community Selection (Default)
The standard mode runs Leiden to detect communities, then uses the perceptron to select the best algorithm for each community independently.

### Mode 1: Full-Graph Adaptive
Skips Leiden community detection entirely. Instead:
1. Computes global graph features
2. Uses perceptron to select the single best algorithm for the entire graph
3. Applies that algorithm

**Use full-graph mode when:**
- Graph has weak community structure (low modularity)
- You want to quickly identify the best single algorithm
- Comparing against per-community selection

### Multi-Level Recursion (max_depth > 0)
When `max_depth > 0`, AdaptiveOrder can recurse into large communities:
1. Detect communities at level 0
2. For communities larger than `min_recurse_size`, run Leiden again
3. Select algorithms for sub-communities
4. Repeat until `max_depth` is reached

```
Level 0:  [   Community A   ]  [Community B]  [C]
                  |
Level 1:  [SubA1]  [SubA2]  [SubA3]
                      |
Level 2:      [SubA2a]  [SubA2b]
```

## Architecture Diagram

```
+------------------+
|   INPUT GRAPH    |
+--------+---------+
         |
         v
+------------------+
| Leiden Community |
|    Detection     |
+--------+---------+
         |
    +----+----+----+----+
    |         |         |
    v         v         v
+-------+ +-------+ +-------+
| Comm1 | | Comm2 | | CommN |
+---+---+ +---+---+ +---+---+
    |         |         |
    v         v         v
+-------+ +-------+ +-------+
|Feature| |Feature| |Feature|
|Extract| |Extract| |Extract|
+---+---+ +---+---+ +---+---+
    |         |         |
    v         v         v
+-------+ +-------+ +-------+
|Percep-| |Percep-| |Percep-|
| tron  | | tron  | | tron  |
|Select | |Select | |Select |
+---+---+ +---+---+ +---+---+
    |         |         |
    v         v         v
 Rabbit    HubClust  LeidenCSR
 Order       DBG
    |         |         |
    +----+----+----+----+
         |
         v
+------------------+
|  MERGE & OUTPUT  |
| (size-sorted)    |
+------------------+
```

## Auto-Clustering Type System

AdaptiveOrder uses automatic clustering to group similar graphs, rather than predefined categories:

**How It Works:**
1. Extract 7 features per graph: modularity, log_nodes, log_edges, avg_degree, degree_variance, hub_concentration, clustering_coefficient
2. Cluster similar graphs using k-means-like clustering
3. Train optimized weights for each cluster
4. At runtime, find best matching cluster based on Euclidean distance to centroids

**Type Files:**
```
scripts/weights/
├── active/               # C++ reads from here
│   ├── type_registry.json
│   ├── type_0.json
│   └── type_N.json
├── merged/               # Accumulated from all runs
└── runs/                 # Historical snapshots
```

**Weight File Loading Priority:**
1. Environment variable `PERCEPTRON_WEIGHTS_FILE` (if set)
2. Best matching type file (e.g., `scripts/weights/active/type_0.json`)
3. Semantic type fallback (if type files don't exist)
4. Hardcoded defaults

## Type Matching at Runtime

Each type has a **centroid** — the average feature vector of its training graphs. At runtime, the system computes the new graph's features and selects the type with the smallest Euclidean distance. If all centroids are too far (distance > 1.5), the OOD guardrail falls back to ORIGINAL.

## Why Per-Community Selection?

Different parts of a graph have different structures:
- **Hub communities**: Dense cores with high-degree vertices → HUBCLUSTERDBG works well
- **Sparse communities**: Mesh-like structures → RCM or ORIGINAL may be better
- **Hierarchical communities**: Tree-like → LeidenCSR traversal variants (`graphbrew:dfs`, `graphbrew:bfs`) excel

AdaptiveOrder selects the best algorithm for each community's characteristics.

## How to Use

### Basic Usage

```bash
# Let AdaptiveOrder choose automatically
./bench/bin/pr -f graph.el -s -o 14 -n 3
```

### Output Explained

```
=== Adaptive Reordering Selection (Depth 0, Modularity: 0.0301) ===
Comm    Nodes   Edges   Density DegVar  HubConc Selected
131     1662    16151   0.0117  1.9441  0.5686  LeidenCSR
272     103     149     0.0284  2.8547  0.4329  Original
489     178     378     0.0240  1.3353  0.3968  HUBCLUSTERDBG
...

=== Algorithm Selection Summary ===
Original: 846 communities
LeidenCSR: 3 communities
HUBCLUSTERDBG: 2 communities
```

This shows:
- Each community's features
- Which algorithm was selected for each
- Summary of selections

---

## The Perceptron Model

### What is a Perceptron?

A **perceptron** is the simplest form of a neural network - a linear classifier that computes a weighted sum of inputs.

**Mathematical Formula:**
```
output = activation(sum(w_i * x_i) + bias)

Where:
  x_i = input features (modularity, density, etc.)
  w_i = learned weights (how important each feature is)
  bias = base score (algorithm's inherent quality)
```

**Why Perceptron for GraphBrew?**
1. **Interpretable**: Each weight tells us feature importance
2. **Fast**: O(n) computation where n = number of features
3. **Online Learning**: Can update weights incrementally
4. **No Overfitting**: Simple model generalizes well

For multi-class selection, we use **one perceptron per algorithm**. Each computes a score, and we pick the algorithm with the **highest score**.

### Perceptron Scoring Diagram

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
2. ORIGINAL Margin: If best - ORIGINAL < 0.05 → keep ORIGINAL
```

### Features Used

The C++ code computes these features for each community at runtime:

#### Linear Features (15)

| Feature | Weight Field | Description | Range |
|---------|--------------|-------------|-------|
| `modularity` | `w_modularity` | Community cohesion | 0.0 - 1.0 |
| `log_nodes` | `w_log_nodes` | log10(num_nodes) | 0 - 10 |
| `log_edges` | `w_log_edges` | log10(num_edges) | 0 - 15 |
| `density` | `w_density` | edges / max_edges | 0.0 - 1.0 |
| `avg_degree` | `w_avg_degree` | mean degree / 100 | 0.0 - 1.0 |
| `degree_variance` | `w_degree_variance` | degree distribution spread | 0.0 - 5.0 |
| `hub_concentration` | `w_hub_concentration` | fraction of edges to top 10% | 0.0 - 1.0 |
| `clustering_coeff` | `w_clustering_coeff` | local clustering (sampled) | 0.0 - 1.0 |
| `avg_path_length` | `w_avg_path_length` | estimated avg path / 10 | 0 - 5.0 |
| `diameter_estimate` | `w_diameter` | estimated diameter / 50 | 0 - 2.0 |
| `community_count` | `w_community_count` | log10(sub-communities) | 0 - 3.0 |
| `reorder_time` | `w_reorder_time` | estimated reorder time | 0 - 100s |
| `packing_factor` | `w_packing_factor` | avg_degree / max_degree (uniformity) | 0.0 - 1.0 |
| `forward_edge_fraction` | `w_forward_edge_fraction` | edges to higher-ID vertices | 0.0 - 1.0 |
| `working_set_ratio` | `w_working_set_ratio` | log₂(graph_bytes / LLC_size + 1) | 0 - 10 |

#### Quadratic Cross-Terms (3)

| Interaction | Weight Field | Description |
|-------------|--------------|-------------|
| degree_variance × hub_concentration | `w_dv_x_hub` | Power-law graph indicator |
| modularity × log₁₀(nodes) | `w_mod_x_logn` | Large modular vs small modular |
| packing_factor × log₂(wsr+1) | `w_pf_x_wsr` | Uniform-degree + cache pressure |

#### Convergence Bonus (1)

| Feature | Weight Field | Description |
|---------|--------------|-------------|
| forward_edge_fraction | `w_fef_convergence` | Added only for PR/SSSP benchmarks |

> **New Features:** `packing_factor` (IISWC'18), `forward_edge_fraction` (GoGraph), and `working_set_ratio` (P-OPT) capture degree uniformity, ordering quality, and cache pressure respectively. The quadratic cross-terms capture non-linear feature interactions.

> **LLC Detection:** The `working_set_ratio` is computed by dividing the graph's memory footprint (offsets + edges + vertex data) by the system's L3 cache size, detected via `GetLLCSizeBytes()` using `sysconf(_SC_LEVEL3_CACHE_SIZE)` on Linux (30 MB fallback).

### C++ Code Architecture

AdaptiveOrder's implementation is split across modular header files in `bench/include/graphbrew/reorder/`:

| File | Purpose |
|------|---------|
| `reorder_types.h` | Base types, perceptron model, `ComputeSampledDegreeFeatures` |
| `reorder_adaptive.h` | `AdaptiveConfig` struct, adaptive selection utilities |

**AdaptiveConfig Struct:**

```cpp
// bench/include/graphbrew/reorder/reorder_adaptive.h
struct AdaptiveConfig {
    int max_depth = 0;           // Recursion depth (0 = per-community only)
    double resolution = 0.0;     // Leiden resolution (0 = auto)
    int min_recurse_size = 50000; // Min nodes for recursion
    int mode = 0;                // 0 = per-community, 1 = full-graph

    static AdaptiveConfig FromOptions(const std::string& options);
    void print() const;
};

// Usage in builder.h:
AdaptiveConfig config = AdaptiveConfig::FromOptions("2:0.75:50000:0");
// → max_depth=2, resolution=0.75, min_recurse_size=50000, mode=0
```

**ComputeSampledDegreeFeatures Utility:**

For fast topology analysis without computing over the entire graph:

```cpp
// bench/include/graphbrew/reorder/reorder_types.h
struct SampledDegreeFeatures {
    double degree_variance;
    double hub_concentration;
    double avg_degree;
    double clustering_coeff;
    double working_set_ratio;  // graph_bytes / LLC_size
};

template<typename GraphT>
SampledDegreeFeatures ComputeSampledDegreeFeatures(
    const GraphT& g, 
    size_t sample_size = 1000
);

// Detects system LLC size via sysconf (Linux) with 30MB fallback
size_t GetLLCSizeBytes();

// Samples ~1000 vertices to estimate graph topology features
// Also computes working_set_ratio using LLC detection
// Used by: GenerateAdaptiveMappingFullGraph, GenerateAdaptiveMappingRecursive,
//          ComputeAndPrintGlobalTopologyFeatures
```

**Key Functions in builder.h:**

```cpp
// Per-community adaptive selection (mode=0)
void GenerateAdaptiveMappingRecursive(
    const CSRGraph& g, pvector<NodeID_>& new_ids,
    const ReorderingOptions& opts, int depth = 0);

// Full-graph adaptive selection (mode=1)
void GenerateAdaptiveMappingFullGraph(
    const CSRGraph& g, pvector<NodeID_>& new_ids,
    const ReorderingOptions& opts);

// Unified entry point
void GenerateAdaptiveMappingUnified(
    const CSRGraph& g, pvector<NodeID_>& new_ids,
    const ReorderingOptions& opts);
```

### Weight Structure

Each algorithm has weights for each feature. See [[Perceptron-Weights#file-structure]] for full JSON format, all weight categories, and tuning strategies.

### Benchmark-Specific Scoring

The perceptron supports per-benchmark multipliers via `benchmark_weights` in each algorithm's weight entry. The final score is `base_score × benchmark_weights[type]`.

```cpp
// C++ Usage:
SelectReorderingPerceptron(features);            // BENCH_GENERIC (multiplier = 1.0)
SelectReorderingPerceptron(features, BENCH_PR);  // PageRank-optimized
SelectReorderingPerceptron(features, "bfs");     // BFS-optimized
```

### Score Calculation (C++ Runtime)

```
base_score = bias
           + w_modularity × modularity
           + w_log_nodes × log10(nodes+1)
           + w_log_edges × log10(edges+1)
           + w_density × density
           + w_avg_degree × avg_degree / 100
           + w_degree_variance × degree_variance
           + w_hub_concentration × hub_concentration
           + w_clustering_coeff × clustering_coeff       (if computed)
           + w_avg_path_length × avg_path_length / 10    (if computed)
           + w_diameter × diameter / 50                  (if computed)
           + w_community_count × log10(count+1)          (if computed)
           + w_packing_factor × packing_factor            # NEW
           + w_forward_edge_fraction × fwd_edge_frac      # NEW
           + w_working_set_ratio × log₂(wsr+1)            # NEW
           + w_dv_x_hub × dv × hub_conc                  # QUADRATIC
           + w_mod_x_logn × mod × logN                   # QUADRATIC
           + w_pf_x_wsr × pf × log₂(wsr+1)               # QUADRATIC
           + w_reorder_time × reorder_time               (if known)

# Convergence bonus (PR/SSSP only)
if benchmark ∈ {PR, SSSP}:
    base_score += w_fef_convergence × forward_edge_fraction

# Final score with benchmark adjustment
final_score = base_score × benchmark_weights[benchmark_type]

# Safety checks (applied after scoring):
# 1. OOD Guardrail: type_distance > 1.5 → return ORIGINAL
# 2. ORIGINAL Margin: best - ORIGINAL < 0.05 → return ORIGINAL
```

---

## Training the Perceptron

### Quick Training Commands

```bash
# One-click: downloads graphs, runs benchmarks, generates weights
python3 scripts/graphbrew_experiment.py --full --size small

# Train from existing benchmark/cache results
python3 scripts/graphbrew_experiment.py --phase weights

# Complete training pipeline
python3 scripts/graphbrew_experiment.py --train --size small

# Iterative training to reach target accuracy
python3 scripts/graphbrew_experiment.py --train-iterative --target-accuracy 80 --size small

# Large-scale batched training
python3 scripts/graphbrew_experiment.py --train-batched --size medium --batch-size 8
```

For consistent benchmarks, use label mapping:
```bash
python3 scripts/graphbrew_experiment.py --generate-maps  # Generate once
python3 scripts/graphbrew_experiment.py --use-maps --phase benchmark  # Reuse
```

See [[Perceptron-Weights]] for the full training pipeline details, gradient update rule, and weight tuning strategies.

### Training Progression (Recommended)

1. **Quick test** on small graphs: `--train-iterative --size small --target-accuracy 75`
2. **Fine-tune** with medium graphs: `--train-iterative --size medium --target-accuracy 80 --learning-rate 0.05`
3. **Validate** on large graphs: `--brute-force --size large`

---

## Safety & Robustness

| Feature | Description |
|---------|-------------|
| **OOD Guardrail** | If graph features are > 1.5 Euclidean distance from all type centroids → return ORIGINAL |
| **ORIGINAL Margin** | If best algorithm's score − ORIGINAL's score < 0.05 → keep ORIGINAL |
| **Convergence Bonus** | For PR/SSSP: adds `w_fef_convergence × forward_edge_fraction` to reward forward-edge-heavy orderings |
| **L2 Regularization** | Weight decay `(1 − 1e-4)` after each gradient update prevents explosion |
| **ORIGINAL Trainable** | ORIGINAL is trained like any algorithm, allowing the model to learn when *not reordering* is optimal |

---

## Cross-Validation

Leave-One-Graph-Out (LOGO) validation measures generalization: hold out one graph, train on the rest, predict the held-out graph, repeat.

```python
from scripts.lib.weights import cross_validate_logo
result = cross_validate_logo(benchmark_results, graph_features, type_registry)
print(f"LOGO: {result['accuracy']:.1%}, Overfit: {result['overfitting_score']:.2f}")
```

| Metric | Good | Concerning |
|--------|------|------------|
| LOGO Accuracy | > 60% | < 40% |
| Overfitting Score | < 0.2 | > 0.3 |
| Full-Train Accuracy | 70-85% | > 95% (likely overfit) |

---

## Advanced Training: `compute_weights_from_results()`

The primary training function in `lib/weights.py` implements a 4-stage pipeline:

1. **Multi-Restart Perceptron Training** — 5 independent perceptrons × 800 epochs per benchmark, z-score normalized features, averaged across restarts and benchmarks
2. **Variant Pre-Collapse** — Only the highest-bias variant per base algorithm is kept (e.g., `GraphBrewOrder_graphbrew:hrab` beats `GraphBrewOrder_graphbrew`)
3. **Regret-Aware Benchmark Multiplier Optimization** — Grid search (30 iterations × 32 log-spaced values) maximizing accuracy while minimizing regret
4. **Save to `type_0.json`** with `_metadata` training statistics

See [[Perceptron-Weights#multi-restart-training--benchmark-multipliers]] for details on the training internals.

### Validation with eval_weights.py

```bash
python3 scripts/eval_weights.py
```

Reports accuracy, median regret, top-2 accuracy, and unique predictions. Current metrics (47 graphs × 4 benchmarks): **46.8% accuracy**, **2.6% median regret**, **64.9% top-2 accuracy**.

### Key Finding: LeidenCSR Dominance

C++ validation on 47 graphs showed that **LeidenCSR** was selected for **99.5% of subcommunities** (8,631 / 8,672). As a single algorithm, it achieves 2.9% median regret — very close to the theoretical best per-community selection. This validates that LeidenCSR is the dominant reordering algorithm for most graph types.

---

## How Correlation Analysis Works

The training pipeline benchmarks all algorithms on diverse graphs, extracts structural features, computes Pearson correlations between features and algorithm performance, and converts correlations to perceptron weights.

See [[Correlation-Analysis]] for the full 5-step process with examples.

---

## How Perceptron Scores Feed Into Ordering

### Complete Example: Ordering a Social Network

For a graph with 10,000 nodes and 5 communities, AdaptiveOrder:

1. **Community Detection** — Leiden finds 5 communities of varying size (600–3,500 nodes)
2. **Feature Extraction** — Computes modularity, hub_concentration, degree_variance, etc. for each community
3. **Perceptron Scoring** — Evaluates all algorithms per community using the score formula above
4. **Algorithm Selection** — e.g., LeidenCSR for hub-heavy communities, ORIGINAL for tiny ones
5. **Per-Community Reordering** — Applies each selected algorithm within its community, producing a unified vertex relabeling

The result: cache-friendly memory layout where hub vertices are clustered together within each community. A large hub-heavy community (hub_concentration=0.62) benefits from LeidenCSR grouping hubs together, while a tiny 600-node community sees negligible improvement — ORIGINAL avoids the overhead.

---

## Debugging

```bash
# Verbose output shows type matching and weight loading
./bench/bin/pr -f graph.el -s -o 14 -n 1 2>&1 | head -50
# Look for: "Best matching type: type_0 (distance: 0.45)"

# Validate weights JSON
python3 -c "import json; json.load(open('scripts/weights/active/type_0.json'))"
```

---

## Performance Considerations

### When AdaptiveOrder Helps Most

✅ Graphs with diverse community structures
✅ Large graphs where wrong algorithm choice is costly
✅ Unknown graphs in automated pipelines

### When to Use Fixed Algorithm Instead

❌ Small graphs (overhead not worth it)
❌ Graphs you know well (just use the best algorithm)
❌ Graphs with uniform structure (all communities similar)

### Overhead

- Leiden community detection: ~5-10% of total time
- Feature computation: ~1-2% of total time
- Perceptron inference: < 1% of total time

Total overhead is usually recovered through better algorithm selection.

---

## Next Steps

- [[Perceptron-Weights]] - Detailed weight file documentation
- [[Correlation-Analysis]] - Understanding the training process
- [[Adding-New-Algorithms]] - Add algorithms to the perceptron

---

[← Back to Home](Home) | [Reordering Algorithms →](Reordering-Algorithms)
