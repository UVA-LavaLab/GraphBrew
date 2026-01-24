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
 Rabbit    HubClust  LeidenDFS
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
1. Extract 9 features per graph: modularity, log_nodes, log_edges, density, avg_degree, degree_variance, hub_concentration, clustering_coefficient, community_count
2. Cluster similar graphs using k-means-like clustering
3. Train optimized weights for each cluster
4. At runtime, find best matching cluster based on Euclidean distance to centroids

**Type Files:**
```
scripts/weights/
├── type_registry.json    # Maps graphs → types + stores centroids
├── type_0.json           # Cluster 0 weights
├── type_1.json           # Cluster 1 weights
└── type_N.json           # Additional clusters as needed
```

**Weight File Loading Priority:**
1. Environment variable `PERCEPTRON_WEIGHTS_FILE` (if set)
2. Best matching type file (e.g., `scripts/weights/type_0.json`)
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
- **Hierarchical communities**: Tree-like → LeidenDFS variants excel

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
INPUTS (Features)         WEIGHTS              OUTPUT
=================         =======              ======

modularity: 0.72  --*---> w_mod: 0.28 ---+
                                         |
density: 0.001    --*---> w_den: -0.15 --+
                                         |
degree_var: 2.1   --*---> w_dv: 0.18 ----+
                                         |
hub_conc: 0.45    --*---> w_hc: 0.22 ----+----> SUM ---> SCORE
                                         |      (+bias)
cluster_coef: 0.3 --*---> w_cc: 0.12 ----+
                                         |
avg_path: 5.2     --*---> w_ap: 0.08 ----+
                                         |
diameter: 16      --*---> w_di: 0.05 ----+
                                         |
...               --*---> ...       -----+


ALGORITHM SELECTION:
====================
RABBITORDER:    score = 2.31  <-- WINNER
LeidenDFS:      score = 2.18
HubClusterDBG:  score = 1.95
GORDER:         score = 1.82
ORIGINAL:       score = 0.50
```

### Features Used (12 Total)

The C++ code computes these features for each community at runtime:

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

### Weight Structure

Each algorithm has weights for each feature. The weights file supports multiple categories:

```json
{
  "LeidenDFS": {
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
           + w_clustering_coeff × clustering_coeff      (if computed)
           + w_avg_path_length × avg_path_length / 10   (if computed)
           + w_diameter × diameter / 50                 (if computed)
           + w_community_count × log10(count+1)         (if computed)
           + w_reorder_time × reorder_time              (if known)

# Final score with benchmark adjustment
final_score = base_score × benchmark_weights[benchmark_type]

# For BENCH_GENERIC (default), multiplier = 1.0, so final_score = base_score
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
python3 scripts/graphbrew_experiment.py --full --download-size SMALL

# Or train from existing benchmark/cache results
python3 scripts/graphbrew_experiment.py --phase weights

# Fill ALL weight fields comprehensively
python3 scripts/graphbrew_experiment.py --fill-weights --graphs small
```

This automatically:
1. Downloads 87 diverse graphs from SuiteSparse (with `--auto-memory`/`--auto-disk` support)
2. Runs all 20 algorithms on each graph
3. Collects cache simulation data (L1/L2/L3 hit rates)
4. Records reorder times for each algorithm
5. Auto-clusters graphs by feature similarity and generates:
   - `scripts/weights/type_registry.json` (graph → type mapping + centroids)
   - `scripts/weights/type_N.json` (per-cluster weights)
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
python3 scripts/graphbrew_experiment.py --train-adaptive --graphs small

# Target 90% accuracy with more iterations
python3 scripts/graphbrew_experiment.py --train-adaptive --target-accuracy 90 --max-iterations 20

# Use slower learning rate for fine-tuning
python3 scripts/graphbrew_experiment.py --train-adaptive --target-accuracy 85 --learning-rate 0.05

# Large-scale training with batching and multi-benchmark support
python3 scripts/graphbrew_experiment.py --train-large --graphs medium --batch-size 8 --train-benchmarks pr bfs cc

# Initialize/upgrade weights with enhanced features before training
python3 scripts/graphbrew_experiment.py --init-weights

# Fill ALL weight fields (cache impacts, topology features, benchmark weights)
python3 scripts/graphbrew_experiment.py --fill-weights --graphs small --max-graphs 5
```

**Note:** Both `--train-adaptive` and `--fill-weights` now use the same type-based weight system (`scripts/weights/type_*.json`).

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
Type weight files: scripts/weights/type_*.json
Training summary saved to: results/training_20250118_123045/training_summary.json
```

### Training Output Files

The training process updates type-based weights and creates:

```
scripts/weights/                # Type-based weights (PRIMARY - updated each iteration)
├── type_registry.json          # Graph → type mapping + centroids
├── type_0.json                 # Cluster 0 weights (updated)
├── type_1.json                 # Cluster 1 weights (updated)
└── type_N.json                 # Additional clusters (updated)

results/training_20250118_123045/
├── training_summary.json       # Overall training results
├── weights_iter1.json          # Legacy snapshot after iteration 1
├── weights_iter2.json          # Legacy snapshot after iteration 2
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
python3 scripts/graphbrew_experiment.py --train-adaptive --graphs small --target-accuracy 75

# Step 2: Fine-tune with medium graphs
python3 scripts/graphbrew_experiment.py --train-adaptive --graphs medium --target-accuracy 80 --learning-rate 0.05

# Step 3: Validate on large graphs
python3 scripts/graphbrew_experiment.py --brute-force --graphs large
```

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
# Quick test with generated graphs
python3 scripts/analysis/correlation_analysis.py --quick
```

### Understanding the Training Output

```
----------------------------------------------------------------------
Computing Perceptron Weights  
----------------------------------------------------------------------
Auto-clustering graphs by feature similarity...
  Created cluster type_0: 5 graphs (mod=0.45, deg_var=0.82, hub=0.35)
  Created cluster type_1: 3 graphs (mod=0.12, deg_var=0.21, hub=0.15)

Weights saved to scripts/weights/type_0.json (5 graphs)
Weights saved to scripts/weights/type_1.json (3 graphs)
Registry saved to scripts/weights/type_registry.json
  C++ will automatically load matching type weights at runtime
```

---

## Weight File Format

### Location

```
GraphBrew/scripts/weights/
├── type_registry.json    # Maps graphs → types + centroids
├── type_0.json           # Cluster 0 weights
├── type_1.json           # Cluster 1 weights
└── type_N.json           # Additional clusters
```

### Structure

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
| 8 | RABBITORDER |
| 9 | GORDER |
| 10 | CORDER |
| 11 | RCM |
| 12 | LeidenOrder |
| 13 | GraphBrewOrder |
| 15 | AdaptiveOrder |
| 16 | LeidenDFS |
| 17 | LeidenDendrogram |
| 18 | LeidenDFSSize |
| 19 | LeidenBFS |
| 20 | LeidenCSR |

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
2. Best matching type file from `scripts/weights/type_N.json`
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
Finding best type match for features: mod=0.4521, deg_var=0.8012, hub=0.3421...
Best matching type: type_0 (similarity: 0.9234)
Loaded 21 weights from scripts/weights/type_0.json
```

### Verify Weights File

```bash
# Check JSON is valid
python3 -c "import json; json.load(open('scripts/weights/type_0.json'))"

# View contents
cat scripts/weights/type_0.json | python3 -m json.tool
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
