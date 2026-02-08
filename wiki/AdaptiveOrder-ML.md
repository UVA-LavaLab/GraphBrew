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

## What is a Centroid?

A **centroid** is the "center point" of a cluster - the average feature values of all graphs in that type.

```
Centroid for type_0 (social networks):
  modularity: 0.72      (high - strong communities)
  degree_variance: 2.1  (high - power-law degrees)
  hub_concentration: 0.45
  avg_degree: 15.3

Centroid for type_1 (road networks):
  modularity: 0.15      (low - mesh-like)
  degree_variance: 0.3  (low - uniform degrees)
  hub_concentration: 0.12
  avg_degree: 2.8
```

## Centroid Matching Diagram

```
FEATURE SPACE (2D simplified view)
==================================

     degree_variance
          ^
          |
     3.0  |        * type_2 (power-law)
          |
     2.5  |    
          |        
     2.0  |  [X] <-- NEW GRAPH        * type_0 (social)
          |         features:             centroid
     1.5  |         mod=0.68
          |         dv=1.9
     1.0  |         hub=0.42
          |
     0.5  |                    * type_1 (road)
          |                      centroid
     0.0  +-----------------------------------> modularity
          0.0  0.2  0.4  0.6  0.8  1.0


DISTANCE CALCULATION:
=====================
dist(X, type_0) = sqrt((0.68-0.72)^2 + (1.9-2.1)^2 + ...) = 0.24
dist(X, type_1) = sqrt((0.68-0.15)^2 + (1.9-0.3)^2 + ...) = 1.82
dist(X, type_2) = sqrt((0.68-0.45)^2 + (1.9-2.8)^2 + ...) = 0.98

RESULT: type_0 (social) is closest --> Load type_0.json
```

## How Type Matching Works at Runtime

When we see a new graph, we compute its features and find the **closest type** using Euclidean distance:

```python
def find_best_type(graph_features, type_registry):
    best_type = None
    min_distance = infinity
    
    for type_name, type_info in type_registry.items():
        centroid = type_info['centroid']
        
        # Euclidean distance in feature space
        distance = sqrt(
            (graph.modularity - centroid.modularity)^2 +
            (graph.degree_variance - centroid.degree_variance)^2 +
            (graph.hub_concentration - centroid.hub_concentration)^2 +
            (graph.avg_degree - centroid.avg_degree)^2
        )
        
        if distance < min_distance:
            min_distance = distance
            best_type = type_name
    
    return best_type  # e.g., "type_3"
```

## Why Per-Community Selection?

Different parts of a graph have different structures:
- **Hub communities**: Dense cores with high-degree vertices → HUBCLUSTERDBG works well
- **Sparse communities**: Mesh-like structures → RCM or ORIGINAL may be better
- **Hierarchical communities**: Tree-like → LeidenDendrogram variants excel

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

Each algorithm has weights for each feature. The weights file supports multiple categories:

```json
{
  "LeidenCSR": {
    "bias": 0.58,
    "w_modularity": 0.0,
    "w_log_nodes": 0.001,
    "w_log_edges": 8.5e-05,
    "w_density": -0.001,
    "w_avg_degree": 0.00017,
    "w_degree_variance": 0.001,
    "w_hub_concentration": 0.001,
    "w_clustering_coeff": 0.0,
    "w_avg_path_length": 0.0,
    "w_diameter": 0.0,
    "w_community_count": 0.0,
    "w_packing_factor": 0.0,
    "w_forward_edge_fraction": 0.0,
    "w_working_set_ratio": 0.0,
    "w_dv_x_hub": 0.0,
    "w_mod_x_logn": 0.0,
    "w_pf_x_wsr": 0.0,
    "w_fef_convergence": 0.0,
    "w_reorder_time": -0.0087,
    "cache_l1_impact": 0,
    "cache_l2_impact": 0,
    "cache_l3_impact": 0,
    "cache_dram_penalty": 0,
    "benchmark_weights": {
      "pr": 1.0,
      "bfs": 1.0,
      "cc": 1.0,
      "sssp": 1.0,
      "bc": 1.0
    },
    "_metadata": {
      "win_rate": 1.0,
      "avg_speedup": 1.17,
      "times_best": 16,
      "sample_count": 16,
      "avg_reorder_time": 8.7,
      "avg_l1_hit_rate": 0.0,
      "avg_l2_hit_rate": 0.0,
      "avg_l3_hit_rate": 0.0
    }
  }
}
```

**Weight Categories:**

| Category | Fields | Usage |
|----------|--------|-------|
| **Core weights** | `bias`, `w_modularity`, `w_density`, `w_degree_variance`, `w_hub_concentration`, `w_log_nodes`, `w_log_edges`, `w_avg_degree` | Used in C++ runtime scoring |
| **Extended graph structure** | `w_clustering_coeff`, `w_avg_path_length`, `w_diameter`, `w_community_count` | Used in C++ runtime if features available |
| **New graph-aware** | `w_packing_factor`, `w_forward_edge_fraction`, `w_working_set_ratio` | Degree uniformity, ordering quality, cache pressure |
| **Quadratic interactions** | `w_dv_x_hub`, `w_mod_x_logn`, `w_pf_x_wsr` | Non-linear feature cross-terms |
| **Convergence** | `w_fef_convergence` | Bonus for PR/SSSP benchmarks only |
| **Reorder time** | `w_reorder_time` | Penalty for slow reordering (used in C++) |
| **Cache impact** | `cache_l1_impact`, `cache_l2_impact`, `cache_l3_impact`, `cache_dram_penalty` | Used during Python training to adjust `bias` |
| **Per-benchmark multipliers** | `benchmark_weights.{pr,bfs,cc,sssp,bc,tc}` | Benchmark-specific score adjustments |
| **Metadata** | `_metadata.*` | Statistics, not used in scoring |

### Benchmark-Specific Weights (NEW)

The perceptron supports **benchmark-specific tuning**. Some algorithms perform differently across workloads:
- **PageRank**: Iterative, benefits from cache locality
- **BFS**: Traversal-heavy, benefits from hub ordering
- **SSSP**: Priority-queue based, different access patterns

**Benchmark Types (C++ Enum):**

```cpp
enum BenchmarkType {
    BENCH_GENERIC = 0,  // Default - balanced for all algorithms
    BENCH_PR,           // PageRank
    BENCH_BFS,          // Breadth-First Search
    BENCH_CC,           // Connected Components
    BENCH_SSSP,         // Single-Source Shortest Path
    BENCH_BC,           // Betweenness Centrality
    BENCH_TC            // Triangle Counting
};
```

**How It Works:**

1. Base score is computed from graph features
2. Score is multiplied by `benchmark_weights[current_benchmark]`
3. If no benchmark is specified, `BENCH_GENERIC` is used (multiplier = 1.0)

```cpp
// C++ Usage Examples:

// Generic/default - optimizes for all algorithms equally
ReorderingAlgo algo = SelectReorderingPerceptron(features);  // BENCH_GENERIC
ReorderingAlgo algo = SelectReorderingPerceptron(features, BENCH_GENERIC);
ReorderingAlgo algo = SelectReorderingPerceptron(features, "generic");

// Benchmark-specific - optimizes for that workload
ReorderingAlgo algo = SelectReorderingPerceptron(features, BENCH_PR);
ReorderingAlgo algo = SelectReorderingPerceptron(features, "pr");
```

> **Note:** The `cache_*` and `benchmark_weights` fields are primarily used during Python training to compute the final `bias` value. At C++ runtime, you can optionally pass a benchmark type to apply the benchmark-specific multiplier.

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

### Score Calculation Example

```
Community features:
- modularity: 0.5
- log_nodes: 4.0
- log_edges: 5.0
- density: 0.01
- avg_degree: 0.2
- degree_variance: 1.5
- hub_concentration: 0.4

LeidenCSR score:
= 0.58                    # bias (incorporates cache/benchmark adjustments)
+ 0.0 × 0.5               # modularity
+ 0.001 × 4.0             # log_nodes
+ 8e-05 × 5.0             # log_edges
+ (-0.001) × 0.01         # density
+ 0.00016 × 0.2           # avg_degree
+ 0.001 × 1.5             # degree_variance
+ 0.001 × 0.4             # hub_concentration
= 0.58 + 0.004 + 0.0004 - 0.00001 + 0.000032 + 0.0015 + 0.0004
≈ 0.586
```

---

## Training the Perceptron

### Automatic Training via Unified Pipeline (Recommended)

The easiest way to train is using the unified experiment script:

```bash
# One-click: downloads graphs, runs benchmarks, generates weights
python3 scripts/graphbrew_experiment.py --full --size small

# Or train from existing benchmark/cache results
python3 scripts/graphbrew_experiment.py --phase weights

# Complete training pipeline
python3 scripts/graphbrew_experiment.py --train --size small
```

This automatically:
1. Downloads 87 diverse graphs from SuiteSparse (with `--auto-memory`/`--auto-disk` support)
2. Runs all 18 algorithms on each graph
3. Collects cache simulation data (L1/L2/L3 hit rates)
4. Records reorder times for each algorithm
5. Auto-clusters graphs by feature similarity and generates:
   - `scripts/weights/active/type_registry.json` (graph → type mapping + centroids)
   - `scripts/weights/active/type_N.json` (per-cluster weights)
   - Core feature weights (modularity, size, hub concentration, etc.)
   - Cache impact weights (L1/L2/L3 bonuses, DRAM penalty)
   - Reorder time penalty weight
   - Training metadata (win rate, avg speedup, sample counts)

### Using Pre-Generated Label Maps

For consistent benchmarks across multiple runs, use label mapping:

```bash
# Generate label maps once (also records reorder times)
python3 scripts/graphbrew_experiment.py --generate-maps

# Use maps for all subsequent benchmarks
python3 scripts/graphbrew_experiment.py --use-maps --phase benchmark
```

This ensures each algorithm's reordering is applied consistently, avoiding variance from regenerating orderings. The `--generate-maps` option also records reorder times to `reorder_times_*.json` and `reorder_times_*.csv` files.

---

## Iterative Training with Feedback Loop

The most powerful way to train AdaptiveOrder is using the iterative feedback loop. This process uses the **type-based weight system**:

1. Measures current accuracy (% of times adaptive picks the best algorithm)
2. Identifies where adaptive made wrong predictions
3. Detects the graph type for each subcommunity (type_0, type_1, etc.)
4. Updates type-specific weights via `update_type_weights_incremental()`
5. Repeats until target accuracy is reached

### Running Iterative Training

```bash
# Basic iterative training (default: 80% accuracy, 10 iterations)
python3 scripts/graphbrew_experiment.py --train-iterative --size small

# Target 90% accuracy with more iterations
python3 scripts/graphbrew_experiment.py --train-iterative --target-accuracy 90 --max-iterations 20

# Use slower learning rate for fine-tuning
python3 scripts/graphbrew_experiment.py --train-iterative --target-accuracy 85 --learning-rate 0.05

# Large-scale training with batching and multi-benchmark support
python3 scripts/graphbrew_experiment.py --train-batched --size medium --batch-size 8 --train-benchmarks pr bfs cc

# Initialize/upgrade weights with enhanced features before training
python3 scripts/graphbrew_experiment.py --init-weights

# Complete training pipeline (cache impacts, topology features, benchmark weights)
python3 scripts/graphbrew_experiment.py --train --size small --max-graphs 5
```

**Note:** Both `--train-iterative` and `--train` now use the same type-based weight system (`scripts/weights/active/type_*.json`). Use `--list-runs` to see historical training runs and `--merge-runs` to consolidate weights.

### Training Output

```
============================================================
TRAINING ITERATION 1/10
============================================================

--- Step 1: Measure Current Accuracy ---
[2025-01-18 12:30:00] INFO: Running brute-force analysis...

Iteration 1 Accuracy:
  Adaptive correct (time): 45.0%
  Adaptive correct (cache): 52.0%
  Adaptive in top 3: 78.0%
  Avg time ratio: 0.856

--- Step 2: Analyze Errors and Adjust Type-Based Weights ---
  Weights updated for 15 algorithm adjustments
  Types updated: 4 (type_0, type_2, type_5, type_8)

============================================================
TRAINING ITERATION 2/10
============================================================

Iteration 2 Accuracy:
  Adaptive correct (time): 58.0%
  Adaptive correct (cache): 61.0%
  ...

============================================================
🎯 TARGET ACCURACY REACHED: 82.0% >= 80.0%
============================================================
TRAINING COMPLETE (TYPE-BASED WEIGHTS)
Iterations run: 4
Target accuracy: 80.0%
Final accuracy (time): 82.0%
Target reached: YES
Best iteration: 4
Total unique types updated: 6
Type weight files: scripts/weights/active/type_*.json
Training summary saved to: results/training_20250118_123045/training_summary.json
```

### Training Output Files

The training process updates type-based weights and creates:

```
scripts/weights/                # Type-based weights directory
├── active/                     # C++ reads from here (PRIMARY)
│   ├── type_registry.json      # Graph → type mapping + centroids
│   ├── type_0.json             # Cluster 0 weights (updated)
│   ├── type_1.json             # Cluster 1 weights (updated)
│   └── type_N.json             # Additional clusters (updated)
├── merged/                     # Accumulated from all runs
└── runs/                       # Historical snapshots
    └── run_YYYYMMDD_*/

results/training_20250118_123045/
├── training_summary.json       # Overall training results
├── weights_iter1.json          # Snapshot after iteration 1
├── weights_iter2.json          # Snapshot after iteration 2
├── ...
├── best_weights_iter4.json     # Best snapshot (highest accuracy)
└── brute_force_analysis_*.json # Detailed analysis per iteration
```

**Type-Based Training:** Each iteration classifies graphs into types and updates the corresponding `type_*.json` files. This enables fine-grained, per-graph-type algorithm selection.

### How the Learning Works

For each graph where adaptive picked the wrong algorithm:

1. **Identify the error**: Adaptive selected `RCM`, but `LeidenCSR` was fastest
2. **Analyze features**: High hub_concentration (0.62), high degree_variance (1.8)
3. **Adjust weights**:
   - Increase `LeidenCSR.w_hub_concentration` (should select it for hub graphs)
   - Increase `LeidenCSR.bias` (slightly more likely to be selected)
   - Decrease `RCM.bias` (was over-selected)

The learning rate controls how aggressive these adjustments are:
- High learning rate (0.2-0.5): Fast learning but may overshoot
- Low learning rate (0.01-0.05): Slow but stable convergence

### Best Practices for Training

1. **Use diverse graphs**: Include graphs of different sizes and structures
2. **Start with SMALL graphs**: Faster iteration for initial training
3. **Graduate to larger graphs**: Fine-tune with MEDIUM/LARGE graphs
4. **Monitor convergence**: If accuracy plateaus, adjust learning rate

```bash
# Recommended training progression:
# Step 1: Quick training on small graphs
python3 scripts/graphbrew_experiment.py --train-iterative --size small --target-accuracy 75

# Step 2: Fine-tune with medium graphs
python3 scripts/graphbrew_experiment.py --train-iterative --size medium --target-accuracy 80 --learning-rate 0.05

# Step 3: Validate on large graphs
python3 scripts/graphbrew_experiment.py --brute-force --size large
```

---

## Safety & Robustness Features

### Out-of-Distribution (OOD) Guardrail

When AdaptiveOrder encounters a graph whose features are very different from all training graphs, it risks making a bad prediction. The OOD guardrail prevents this:

```
UNKNOWN_TYPE_DISTANCE_THRESHOLD = 1.5

if euclidean_distance(graph_features, nearest_centroid) > 1.5:
    return ORIGINAL   # Safe fallback - don't risk a bad prediction
```

**Why 1.5?** The type-matching uses 7 features, all normalized to [0, 1]. The maximum possible Euclidean distance in 7D space is √7 ≈ 2.65, so 1.5 represents a moderately far point — the graph is genuinely unlike anything the model has seen.

**Exceptions:** `MODE_FASTEST_REORDER` bypasses the OOD check since it optimizes for reordering speed, not algorithm quality.

### ORIGINAL Margin Fallback

If the perceptron's best-scoring algorithm barely beats ORIGINAL, the system keeps ORIGINAL to avoid reordering overhead for marginal gains:

```
ORIGINAL_MARGIN_THRESHOLD = 0.05

if best_score - original_score < 0.05:
    return ORIGINAL   # Gain too small to justify reordering
```

### Convergence-Aware Scoring

For iterative algorithms like **PageRank** and **SSSP**, edge direction significantly impacts convergence speed. AdaptiveOrder adds a convergence bonus:

```cpp
// In score() method (not scoreBase):
if (benchmark == BENCH_PR || benchmark == BENCH_SSSP) {
    score += w_fef_convergence * forward_edge_fraction;
}
```

This allows the model to learn that certain orderings (with more forward edges) converge faster for iterative workloads, without affecting non-iterative benchmarks like BFS or CC.

### L2 Regularization (Weight Decay)

To prevent weight explosion during long training runs, all feature weights undergo L2 decay after each gradient update:

```python
WEIGHT_DECAY = 1e-4

# After gradient updates:
for key in weights:
    if key.startswith('w_') or key.startswith('cache_'):
        weights[key] *= (1.0 - WEIGHT_DECAY)
```

This keeps weights bounded and improves generalization to unseen graphs.

### ORIGINAL as a Trainable Algorithm

ORIGINAL is now trained like any other algorithm in the correlation-based weight system. This allows the perceptron to learn when *not reordering* is optimal — for example, on small graphs, already well-ordered graphs, or graphs with very weak community structure.

---

## Cross-Validation

### Leave-One-Graph-Out (LOGO) Validation

To measure generalization quality, use LOGO cross-validation:

```bash
# Via Python
python3 -c "
from scripts.lib.weights import cross_validate_logo
# ... load benchmark_results, graph_features, type_registry ...
result = cross_validate_logo(benchmark_results, graph_features, type_registry)
print(f'LOGO accuracy: {result[\"accuracy\"]:.1%}')
print(f'Overfitting score: {result[\"overfitting_score\"]:.2f}')
"
```

**Process:**
1. Hold out one graph
2. Train weights on all remaining graphs
3. Predict the best algorithm for the held-out graph
4. Compare to actual best → correct/incorrect
5. Repeat for every graph

**Interpreting Results:**
| Metric | Good | Concerning |
|--------|------|------------|
| LOGO Accuracy | > 60% | < 40% |
| Overfitting Score | < 0.2 | > 0.3 |
| Full-Train Accuracy | 70-85% | > 95% (likely overfit) |

---

## How Correlation Analysis Works

### Step 1: Benchmark All Algorithms

For each graph in your dataset, we run every algorithm and measure execution time:

```
Graph: facebook.el
┌────────────────────┬──────────────┬─────────┐
│ Algorithm          │ Time (sec)   │ Rank    │
├────────────────────┼──────────────┼─────────┤
│ ORIGINAL (0)       │ 0.0523       │ 5       │
│ HUBCLUSTERDBG (7)  │ 0.0412       │ 3       │
│ LeidenOrder (15)   │ 0.0398       │ 2       │
│ LeidenCSR (17)     │ 0.0371       │ 1 ★     │
│ RCM (11)           │ 0.0489       │ 4       │
└────────────────────┴──────────────┴─────────┘
Winner: LeidenCSR (fastest)
```

### Step 2: Extract Graph Features

For each graph, we compute structural features:

```
Graph: facebook.el
┌─────────────────────┬────────────┐
│ Feature             │ Value      │
├─────────────────────┼────────────┤
│ num_nodes           │ 4,039      │
│ num_edges           │ 88,234     │
│ modularity          │ 0.835      │  ← High (strong communities)
│ density             │ 0.0108     │  ← Sparse
│ avg_degree          │ 43.7       │
│ degree_variance     │ 52.4       │  ← High (hubs exist)
│ hub_concentration   │ 0.42       │  ← Moderate hub dominance
└─────────────────────┴────────────┘
```

### Step 3: Build the Correlation Table

We create a table linking features to winning algorithms:

```
┌──────────────────┬────────────┬─────────┬─────────┬─────────┬────────────────┐
│ Graph            │ Modularity │ Density │ HubConc │ DegVar  │ Best Algorithm │
├──────────────────┼────────────┼─────────┼─────────┼─────────┼────────────────┤
│ facebook.el      │ 0.835      │ 0.011   │ 0.42    │ 52.4    │ LeidenCSR   │
│ twitter.el       │ 0.721      │ 0.002   │ 0.68    │ 891.2   │ LeidenDendrogram   │
│ roadNet-CA.el    │ 0.112      │ 0.0001  │ 0.05    │ 1.2     │ RCM            │
│ web-Google.el    │ 0.654      │ 0.008   │ 0.55    │ 234.5   │ HUBCLUSTERDBG  │
│ citation.el      │ 0.443      │ 0.003   │ 0.31    │ 45.6    │ LeidenOrder    │
└──────────────────┴────────────┴─────────┴─────────┴─────────┴────────────────┘
```

### Step 4: Compute Correlations

We calculate Pearson correlation between each feature and algorithm performance:

```
Feature Correlations with "LeidenCSR being best":
┌─────────────────────┬─────────────┬────────────────────────────────────┐
│ Feature             │ Correlation │ Interpretation                     │
├─────────────────────┼─────────────┼────────────────────────────────────┤
│ modularity          │ +0.78       │ Strong positive → use on modular   │
│ hub_concentration   │ +0.45       │ Moderate positive → helps with hubs│
│ density             │ -0.23       │ Weak negative → prefers sparse     │
│ degree_variance     │ +0.52       │ Moderate positive → handles skew   │
└─────────────────────┴─────────────┴────────────────────────────────────┘
```

### Step 5: Convert Correlations to Perceptron Weights

The correlations become the perceptron weights:

```python
# Simplified weight derivation
weights["LeidenCSR"] = {
    "bias": 0.5 + (win_rate * 0.5),      # Base preference from win rate
    "w_modularity": correlation_modularity * scale,      # +0.78 → +0.25
    "w_hub_concentration": correlation_hubconc * scale,  # +0.45 → +0.15
    "w_density": correlation_density * scale,            # -0.23 → -0.08
    "w_degree_variance": correlation_degvar * scale,     # +0.52 → +0.17
}
```

The scaling factor normalizes correlations to useful weight ranges (typically 0.1-0.3).

---

## How Perceptron Scores Feed Into Ordering

### Complete Example: Ordering a Social Network

Let's walk through ordering `social_network.el` with AdaptiveOrder:

#### Phase 1: Community Detection

```
Input Graph: 10,000 nodes, 150,000 edges

Leiden detects 5 communities:
┌─────────┬───────┬────────┬──────────┬─────────┬──────────┐
│ Comm ID │ Nodes │ Edges  │ Density  │ HubConc │ DegVar   │
├─────────┼───────┼────────┼──────────┼─────────┼──────────┤
│ C0      │ 3,500 │ 45,000 │ 0.0073   │ 0.62    │ 89.3     │
│ C1      │ 2,800 │ 28,000 │ 0.0071   │ 0.41    │ 34.2     │
│ C2      │ 1,900 │ 22,000 │ 0.0122   │ 0.28    │ 12.1     │
│ C3      │ 1,200 │ 8,500  │ 0.0118   │ 0.15    │ 5.4      │
│ C4      │ 600   │ 3,200  │ 0.0178   │ 0.09    │ 2.1      │
└─────────┴───────┴────────┴──────────┴─────────┴──────────┘
```

#### Phase 2: Feature Computation for Each Community

For community C0:
```
Features (normalized):
- modularity: 0.72 (from Leiden quality)
- log_nodes: 3.54 (log10(3500))
- log_edges: 4.65 (log10(45000))
- density: 0.0073
- avg_degree: 25.7 / 100 = 0.257
- degree_variance: 89.3 / 100 = 0.893
- hub_concentration: 0.62
```

#### Phase 3: Perceptron Scoring for Community C0

```
Algorithm Scores for C0 (high hubs, high variance):

LeidenCSR:
  = 0.85 + (0.25×0.72) + (0.1×3.54) + (0.1×4.65) + (-0.05×0.007)
    + (0.15×0.257) + (0.15×0.893) + (0.25×0.62)
  = 0.85 + 0.18 + 0.354 + 0.465 - 0.0004 + 0.039 + 0.134 + 0.155
  = 2.18 ★ WINNER

HUBCLUSTERDBG:
  = 0.75 + (0.1×0.72) + (0.05×3.54) + (0.05×4.65) + (0.0×0.007)
    + (0.2×0.257) + (0.2×0.893) + (0.35×0.62)
  = 0.75 + 0.072 + 0.177 + 0.233 + 0 + 0.051 + 0.179 + 0.217
  = 1.68

ORIGINAL:
  = 0.6 + (0.0×0.72) + (-0.1×3.54) + (-0.1×4.65) + (0.1×0.007)
    + (0.0×0.257) + (0.0×0.893) + (0.0×0.62)
  = 0.6 + 0 - 0.354 - 0.465 + 0.0007 + 0 + 0 + 0
  = -0.22

RCM:
  = 0.65 + (0.05×0.72) + (0.0×3.54) + (0.0×4.65) + (0.15×0.007)
    + (0.1×0.257) + (-0.1×0.893) + (-0.05×0.62)
  = 0.65 + 0.036 + 0 + 0 + 0.001 + 0.026 - 0.089 - 0.031
  = 0.59
```

**Result**: C0 gets **LeidenCSR** (score 2.18)

#### Phase 4: Scoring All Communities

```
┌─────────┬──────────────────┬────────────────────────────────────────┐
│ Comm    │ Selected Algo    │ Reasoning                              │
├─────────┼──────────────────┼────────────────────────────────────────┤
│ C0      │ LeidenCSR     │ High hub_conc (0.62), high deg_var     │
│ C1      │ LeidenCSR     │ Moderate hub_conc, good modularity     │
│ C2      │ LeidenOrder      │ Lower hub_conc, still modular          │
│ C3      │ HUBCLUSTERDBG    │ Small community, moderate structure    │
│ C4      │ ORIGINAL         │ Very small, overhead not worth it      │
└─────────┴──────────────────┴────────────────────────────────────────┘
```

#### Phase 5: Apply Per-Community Ordering

```
Final Vertex Relabeling:

Original IDs → New IDs (after per-community reordering)

Community C0 (LeidenCSR applied):
  Vertices 0-3499 → reordered by hub-aware DFS within C0
  New IDs: 0-3499

Community C1 (LeidenCSR applied):
  Vertices 3500-6299 → reordered by hub-aware DFS within C1
  New IDs: 3500-6299

Community C2 (LeidenOrder applied):
  Vertices 6300-8199 → simple community ordering
  New IDs: 6300-8199

Community C3 (HUBCLUSTERDBG applied):
  Vertices 8200-9399 → hub clustering with DBG
  New IDs: 8200-9399

Community C4 (ORIGINAL - no change):
  Vertices 9400-9999 → kept as-is
  New IDs: 9400-9999
```

#### Visual Representation

```
Before AdaptiveOrder:
┌─────────────────────────────────────────────────────┐
│ Memory layout: vertices scattered, poor locality    │
│ [v892][v12][v5601][v234][v8923][v45]...            │
└─────────────────────────────────────────────────────┘

After AdaptiveOrder:
┌─────────────────────────────────────────────────────┐
│ C0 (LeidenCSR)    │ C1 (LeidenCSR)    │ ...  │
│ [hub1][hub2][n1][n2] │ [hub1][n1][n2][n3]   │      │
│ Hubs clustered first │ Hubs clustered first │      │
└─────────────────────────────────────────────────────┘

Result: When PageRank processes C0, all hub vertices
        are in adjacent cache lines → fewer cache misses
```

---

## Why Different Algorithms for Different Communities?

### The Key Insight

Not all communities have the same structure:

```
Community C0: Social cluster (influencers + followers)
  → High hub concentration (0.62)
  → LeidenCSR groups influencers together
  → Their followers are adjacent in memory

Community C4: Small tight-knit group
  → Low hub concentration (0.09)
  → Few vertices (600)
  → Reordering overhead > benefit
  → ORIGINAL keeps natural ordering
```

### Performance Impact

```
PageRank on Community C0:
┌────────────────────┬──────────────┬─────────────────┐
│ Algorithm          │ Cache Misses │ Time (ms)       │
├────────────────────┼──────────────┼─────────────────┤
│ ORIGINAL           │ 145,000      │ 12.3            │
│ HUBCLUSTERDBG      │ 98,000       │ 8.7             │
│ LeidenCSR       │ 67,000       │ 6.2 ★           │
└────────────────────┴──────────────┴─────────────────┘

PageRank on Community C4:
┌────────────────────┬──────────────┬─────────────────┐
│ Algorithm          │ Cache Misses │ Time (ms)       │
├────────────────────┼──────────────┼─────────────────┤
│ ORIGINAL           │ 1,200        │ 0.4 ★           │
│ LeidenCSR       │ 1,150        │ 0.5 (+ overhead)│
└────────────────────┴──────────────┴─────────────────┘

AdaptiveOrder picks the best for EACH community!
```

---

### Quick Training (Synthetic Graphs)

```bash
# Quick test with small graphs
python3 scripts/graphbrew_experiment.py --full --size small --skip-cache
```

### Understanding the Training Output

```
----------------------------------------------------------------------
Computing Perceptron Weights  
----------------------------------------------------------------------
Auto-clustering graphs by feature similarity...
  Created cluster type_0: 5 graphs (mod=0.45, deg_var=0.82, hub=0.35)
  Created cluster type_1: 3 graphs (mod=0.12, deg_var=0.21, hub=0.15)

Weights saved to scripts/weights/active/type_0.json (5 graphs)
Weights saved to scripts/weights/active/type_1.json (3 graphs)
Registry saved to scripts/weights/active/type_registry.json
  C++ will automatically load matching type weights at runtime
```

---

## Weight File Format

### Location

```
GraphBrew/scripts/weights/
├── active/               # C++ reads from here
│   ├── type_registry.json
│   ├── type_0.json
│   └── type_N.json
├── merged/               # Merged weights from all runs
└── runs/                 # Historical snapshots
```

### Structure

Each weight file contains algorithm-specific weights. The C++ code reads the core weights; additional metadata is used by the Python training system:

```json
{
  "SORT": {
    "bias": 0.5,
    "w_modularity": 0.00018,
    "w_log_nodes": 6.5e-06,
    "w_log_edges": 6.5e-06,
    "w_density": 0.0,
    "w_avg_degree": 1.2e-05,
    "w_degree_variance": 0.00031,
    "w_hub_concentration": 9.7e-05,
    "w_clustering_coeff": 0.00017,
    "w_avg_path_length": 9.2e-05,
    "w_diameter": 4.3e-05,
    "w_community_count": 2.2e-07,
    "w_reorder_time": -6.9e-07,
    "cache_l1_impact": 0.00021,
    "cache_l2_impact": 0.00019,
    "cache_l3_impact": 0.00021,
    "cache_dram_penalty": -7.9e-05,
    "benchmark_weights": {
      "pr": 1.0, "bfs": 1.0, "cc": 1.0, "sssp": 1.0, "bc": 1.0, "tc": 1.0
    },
    "_metadata": {
      "sample_count": 1,
      "avg_speedup": 1.52,
      "win_count": 1,
      "win_rate": 1.0,
      "last_updated": "2026-01-25T01:11:16"
    }
  },
  "LeidenCSR": {
    "bias": 0.85,
    "w_modularity": 0.25,
    ...
  },
  ...
}
```

### Algorithm Name Mapping

| ID | Name in JSON |
|----|--------------|
| 0 | ORIGINAL |
| 1 | RANDOM |
| 2 | SORT |
| 3 | HUBSORT |
| 4 | HUBCLUSTER |
| 5 | DBG |
| 6 | HUBSORTDBG |
| 7 | HUBCLUSTERDBG |
| 8 | RABBITORDER (has variants) |
| 9 | GORDER |
| 10 | CORDER |
| 11 | RCM |
| 12 | GraphBrewOrder (has variants) |
| 13 | MAP |
| 14 | AdaptiveOrder |
| 15 | LeidenOrder |
| 16 | LeidenDendrogram (has variants) |
| 17 | LeidenCSR (has variants) |

> **Note:** For current variant lists, see `scripts/lib/utils.py` which defines:
> `RABBITORDER_VARIANTS`, `GRAPHBREW_VARIANTS`, `LEIDEN_DENDROGRAM_VARIANTS`, `LEIDEN_CSR_VARIANTS`

---

## Manual Weight Tuning

### When to Tune Manually

- You have domain knowledge about your graphs
- Automated training doesn't have enough data
- You want to favor certain algorithms

### Tuning Guidelines

**Bias**: Base preference for algorithm (0.3 - 1.0)
- Higher = algorithm selected more often
- Start around 0.5-0.7

**w_modularity**: How algorithm performs on modular graphs
- Positive = better on high-modularity graphs
- Community algorithms should have positive values (0.1 - 0.3)

**w_log_nodes / w_log_edges**: Scale effects
- Positive = better on larger graphs
- Negative = better on smaller graphs

**w_density**: Sparse vs dense preference
- Positive = better on dense graphs
- Negative = better on sparse graphs

**w_hub_concentration**: Hub structure preference
- Positive = better when hubs dominate
- Hub algorithms (HUBSORT, HUBCLUSTER) should have positive values

### Example: Favoring LeidenCSR for Large Graphs

```json
{
  "LeidenCSR": {
    "bias": 0.9,
    "w_modularity": 0.3,
    "w_log_nodes": 0.2,
    "w_log_edges": 0.2,
    "w_density": -0.1,
    "w_avg_degree": 0.15,
    "w_degree_variance": 0.15,
    "w_hub_concentration": 0.25
  }
}
```

---

## How C++ Loads Weights

### Automatic Loading

The C++ code automatically loads weights from:
1. `PERCEPTRON_WEIGHTS_FILE` environment variable (if set)
2. Best matching type file from `scripts/weights/active/type_N.json`
3. Semantic type fallback (if type files don't exist)
4. Hardcoded defaults (if all else fails)

### Environment Variable Override

```bash
# Use custom weights file
export PERCEPTRON_WEIGHTS_FILE=/path/to/my_weights.json
./bench/bin/pr -f graph.el -s -o 14 -n 3
```

### Fallback to Defaults

If no type weights file matches or exists:
- C++ uses hardcoded defaults
- Warning is printed (in verbose mode)
- System continues to work

---

## Recursive AdaptiveOrder

AdaptiveOrder can recursively apply itself to large communities:

```
Level 0: Full graph → 10 communities
Level 1: Large community → 5 sub-communities
Level 2: Sub-community → final ordering
```

This creates a hierarchical ordering that respects structure at multiple scales.

### Controlling Recursion

The recursion depth is controlled by:
- Community size threshold (min nodes for recursion)
- Maximum depth limit

---

## Debugging AdaptiveOrder

### Verbose Output

```bash
# See detailed selection process
./bench/bin/pr -f graph.el -s -o 14 -n 1 2>&1 | head -50
```

### Check Which Weights Are Loaded

Look for output like:
```
Best matching type: type_0 (distance: 0.4521)
Perceptron: Loaded 5 weights from scripts/weights/active/type_0.json
```

### Verify Weights File

```bash
# Check JSON is valid
python3 -c "import json; json.load(open('scripts/weights/active/type_0.json'))"

# View contents
cat scripts/weights/active/type_0.json | python3 -m json.tool
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
