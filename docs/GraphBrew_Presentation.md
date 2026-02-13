# GraphBrew: Adaptive Graph Reordering with Machine Learning

---

## Slide 1: Title

# **GraphBrew**
## Adaptive Graph Reordering for Cache-Efficient Graph Processing

### Key Innovation: Per-Community Algorithm Selection using Perceptron-based ML

---

## Slide 2: Problem Statement

# Why Graph Reordering Matters

```
┌─────────────────────────────────────────────────────────────────┐
│  Graph Algorithms (BFS, PageRank, SSSP, etc.)                   │
│  ↓                                                              │
│  Random Memory Access Patterns → POOR CACHE UTILIZATION         │
│  ↓                                                              │
│  Up to 70% of execution time waiting for memory!                │
└─────────────────────────────────────────────────────────────────┘
```

### The Challenge:
- **Different graphs** benefit from **different reordering algorithms**
- Social networks → community-based reordering (RabbitOrder, Leiden)
- Road networks → bandwidth reduction (RCM)
- Web graphs → hub-based reordering (HubSort, DBG)

### No Single Algorithm Wins for All Graphs!

---

## Slide 3: GraphBrew Overview

# GraphBrew Architecture

```
                    ┌─────────────────────────────┐
                    │       INPUT GRAPH           │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Leiden Community Detection │
                    │  (Modularity Optimization)  │
                    └──────────────┬──────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Community 1    │     │  Community 2    │     │  Community N    │
│  Features:      │     │  Features:      │     │  Features:      │
│  - density      │     │  - density      │     │  - density      │
│  - hub_conc     │     │  - hub_conc     │     │  - hub_conc     │
│  - deg_var      │     │  - deg_var      │     │  - deg_var      │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PERCEPTRON    │     │   PERCEPTRON    │     │   PERCEPTRON    │
│   ML SELECTOR   │     │   ML SELECTOR   │     │   ML SELECTOR   │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
    RabbitOrder             HubClusterDBG          GraphBrewOrder
```

---

## Slide 4: Feature Extraction

# Graph & Community Features (12 Features)

| Feature | Description | Normalization | Impact |
|---------|-------------|---------------|--------|
| **modularity** | Community structure strength | [0, 1] | High → Leiden algorithms |
| **density** | Edge density | edges/(n*(n-1)/2) | Dense → simple reordering |
| **degree_variance** | CV of degree distribution | σ/μ | High → hub-based methods |
| **hub_concentration** | Top 10% edge share | [0, 1] | High → HubSort variants |
| **clustering_coeff** | Triangle density | [0, 1] | High → local methods |
| **avg_path_length** | Average shortest path | ÷10 | High → BFS-based |
| **diameter** | Graph diameter estimate | ÷50 | Wide → bandwidth reduction |
| **community_count** | Sub-community count | log10 | Many → recursive approach |
| **log_nodes** | Graph size (nodes) | log10 | Scaling factor |
| **log_edges** | Graph size (edges) | log10 | Scaling factor |
| **avg_degree** | Average degree | ÷100 | Connectivity measure |
| **reorder_time** | Algorithm runtime | negative | Time penalty |

---

## Slide 5: Perceptron Model

# ML-Based Algorithm Selection

## Score Computation (Per Algorithm)

```python
score = bias
      + w_modularity × modularity
      + w_log_nodes × log₁₀(nodes) × 0.1
      + w_log_edges × log₁₀(edges) × 0.1
      + w_density × density
      + w_avg_degree × (avg_degree / 100)
      + w_degree_variance × degree_variance
      + w_hub_concentration × hub_concentration
      + w_clustering_coeff × clustering_coeff
      + w_avg_path_length × (avg_path_length / 10)
      + w_diameter × (diameter / 50)
      + w_community_count × log₁₀(community_count)
      + w_reorder_time × reorder_time
      
# Apply benchmark-specific multiplier
final_score = score × benchmark_weights[benchmark]
```

## Algorithm with **highest score** is selected!

---

## Slide 6: Supported Algorithms

# 16 Reordering Algorithms

| ID | Algorithm | Best For |
|----|-----------|----------|
| 0 | ORIGINAL | No reordering (baseline) |
| 1 | RANDOM | Random permutation |
| 2 | SORT | Degree-sorted ordering |
| 3 | HUBSORT | Hub vertices first |
| 4 | HUBCLUSTER | Hub clustering |
| 5 | DBG | Degree-based grouping |
| 6 | HUBSORTDBG | HubSort + DBG hybrid |
| 7 | HUBCLUSTERDBG | HubCluster + DBG |
| 8 | RABBITORDER | Community-aware (csr/boost variants) |
| 9 | GORDER | Graph reordering |
| 10 | CORDER | Cache-optimized |
| 11 | RCM | Reverse Cuthill-McKee (road) |
| 12 | GraphBrewOrder | Per-community reordering |
| 13 | MAP | External mapping file |
| 14 | AdaptiveOrder | ML-powered selection |
| 15 | LeidenOrder | GVE-Leiden baseline |
| ~~16~~ | ~~LeidenCSR~~ | _Deprecated — subsumed by GraphBrewOrder (12)_ |

---

## Slide 7: Weight Training Process

# How We Train the Weights

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                               │
└─────────────────────────────────────────────────────────────────────┘

  Phase 1: Generate Reorderings
  ┌─────────────────────────────────────────────────────────────────┐
  │  For each graph × algorithm:                                    │
  │    1. Run converter to generate reordered graph                 │
  │    2. Record reorder_time                                       │
  │    3. Save .lo mapping file                                     │
  └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  Phase 2: Run Benchmarks
  ┌─────────────────────────────────────────────────────────────────┐
  │  For each graph × algorithm × benchmark (pr, bfs, cc, sssp, bc):│
  │    1. Run benchmark, record trial_time                          │
  │    2. Compute speedup = baseline_time / trial_time              │
  │    3. Extract topology features from C++ output:                │
  │       - Clustering Coefficient, Avg Path Length, Diameter       │
  │       - Degree Variance, Hub Concentration, Community Count     │
  │    4. IMMEDIATELY update weights (incremental learning)         │
  └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  Phase 3: Cache Simulations (Optional)
  ┌─────────────────────────────────────────────────────────────────┐
  │  Simulate L1/L2/L3 cache behavior                               │
  │  Update cache_l1_impact, cache_l2_impact, cache_dram_penalty    │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Slide 8: Gradient Update Rule

# Online Learning with Perceptron Updates

## Error Signal Computation
```python
error = (speedup - 1.0) - current_score
# Positive error: algorithm performed better than expected
# Negative error: algorithm performed worse than expected
```

## Weight Update (Stochastic Gradient Descent)
```python
learning_rate = 0.01

# Core feature weights
w_modularity     += lr × error × modularity
w_density        += lr × error × density  
w_degree_variance += lr × error × degree_variance
w_hub_concentration += lr × error × hub_concentration

# Extended topology weights
w_avg_path_length += lr × error × (avg_path_length / 10)
w_diameter        += lr × error × (diameter / 50)
w_clustering_coeff += lr × error × clustering_coeff
w_community_count += lr × error × (community_count / 1000)

# Cache impact weights (when simulation data available)
w_cache_l1_impact += lr × error × l1_hit_rate
w_cache_dram_penalty += lr × error × dram_penalty
```

---

## Slide 9: Type-Based Weight System

# Graph Type Clustering

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TYPE REGISTRY                                    │
│                  (type_registry.json)                               │
├─────────────────────────────────────────────────────────────────────┤
│  type_0:  centroid = [mod=0.71, deg_var=1.2, hub=0.35, ...]        │
│  type_1:  centroid = [mod=0.45, deg_var=2.8, hub=0.62, ...]        │
│  type_2:  centroid = [mod=0.23, deg_var=0.9, hub=0.18, ...]        │
│  ...                                                                │
│  type_28: centroid = [mod=0.89, deg_var=0.5, hub=0.25, ...]        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  At runtime: compute graph features → find nearest centroid         │
│                                                                     │
│  distance = √[(mod - centroid.mod)² + (deg_var - centroid.deg)² +  │
│               (hub - centroid.hub)² + ...]                          │
│                                                                     │
│  Load weights from type_N.json for best matching type               │
└─────────────────────────────────────────────────────────────────────┘
```

### Benefits:
- **Specialized weights** for different graph families
- **Auto-clustering** discovers natural graph categories
- **Incremental updates** per type file

---

## Slide 10: AdaptiveOrder Algorithm (C++)

# Runtime Algorithm Selection

```cpp
// For each community detected by Leiden:
void GenerateAdaptiveMappingRecursive(...) {
    
    // Step 1: Leiden community detection
    auto communities = runLeiden(graph, resolution=0.75);
    
    // Step 2: For each community, compute features
    for (community : communities) {
        CommunityFeatures feat = ComputeCommunityFeatures(community);
        
        // Features computed:
        // - internal_density, degree_variance, hub_concentration
        // - clustering_coeff, avg_path_length, diameter_estimate
        // - community_count (sub-communities)
        
        // Step 3: Select best algorithm using perceptron
        ReorderingAlgo best = SelectBestReorderingForCommunity(
            feat, global_modularity, ...);
            
        // Step 4: Apply selected algorithm to this community
        ApplyLocalReordering(community, best);
    }
    
    // Step 5: Combine communities in size-sorted order
    AssignGlobalIds(communities_sorted_by_size);
}
```

---

## Slide 11: Feature Computation (C++)

# ComputeCommunityFeatures Implementation

```cpp
CommunityFeatures ComputeCommunityFeatures(
    const vector<NodeID>& comm_nodes,
    const Graph& g) {
    
    CommunityFeatures feat;
    
    // Basic features
    feat.num_nodes = comm_nodes.size();
    feat.num_edges = count_internal_edges(comm_nodes, g);
    feat.internal_density = 2.0 * feat.num_edges / 
                           (feat.num_nodes * (feat.num_nodes - 1));
    
    // Degree statistics
    vector<int64_t> degrees = compute_internal_degrees(comm_nodes, g);
    feat.avg_degree = mean(degrees);
    feat.degree_variance = cv(degrees);  // coefficient of variation
    
    // Hub concentration: top 10% edge share
    sort(degrees, descending);
    feat.hub_concentration = sum(top_10_percent) / sum(all);
    
    // Clustering coefficient (sampled for efficiency)
    feat.clustering_coeff = sampled_triangle_ratio(comm_nodes, g, samples=50);
    
    // Diameter & avg path (single BFS from hub node)
    auto [diameter, avg_path] = bfs_from_hub(comm_nodes, g);
    feat.diameter_estimate = diameter;
    feat.avg_path_length = avg_path;
    
    return feat;
}
```

---

## Slide 12: Benchmark Workloads

# Supported Graph Algorithms

| Benchmark | Algorithm | Access Pattern |
|-----------|-----------|----------------|
| **PR** | PageRank | Pull-based iteration |
| **BFS** | Breadth-First Search | Level-synchronous |
| **CC** | Connected Components | Label propagation |
| **SSSP** | Single-Source Shortest Path | Delta-stepping |
| **BC** | Betweenness Centrality | Multiple BFS |
| **TC** | Triangle Counting | Intersection-based |

### Benchmark-Specific Weight Adjustments
```python
benchmark_weights = {
    'pr': 1.2,    # PageRank benefits most from reordering
    'bfs': 0.9,   # BFS has more sequential access
    'cc': 1.0,    # Neutral
    'sssp': 1.1,  # Similar to BFS but iterative
    'bc': 0.8     # Multiple traversals, less locality gain
}
```

---

## Slide 13: Example Weight File

# type_0.json (Social Network Type)

```json
{
  "RABBITORDER": {
    "bias": 0.35,
    "w_modularity": 0.28,
    "w_log_nodes": 0.05,
    "w_log_edges": 0.03,
    "w_density": -0.15,
    "w_avg_degree": 0.02,
    "w_degree_variance": 0.18,
    "w_hub_concentration": 0.22,
    "w_clustering_coeff": 0.12,
    "w_avg_path_length": 0.08,
    "w_diameter": 0.05,
    "w_community_count": 0.10,
    "w_reorder_time": -0.02,
    "benchmark_weights": {
      "pr": 1.15, "bfs": 1.05, "cc": 1.0, "sssp": 1.1, "bc": 0.9
    },
    "_metadata": {
      "sample_count": 1247,
      "avg_speedup": 1.32,
      "win_rate": 0.68
    }
  },
  "HubClusterDBG": { ... }
}
```

---

## Slide 14: Training Pipeline

# Full Training Workflow

```bash
# 1. Initialize fresh weights (creates type_0.json through type_N.json)
python3 scripts/graphbrew_experiment.py --init-weights

# 2. Fill weights with reasonable defaults based on heuristics
python3 scripts/graphbrew_experiment.py --fill-weights \
    --graphs medium --benchmarks pr bfs cc

# 3. Run comprehensive training
python3 scripts/graphbrew_experiment.py --train-adaptive \
    --graphs all \
    --benchmarks pr bfs cc sssp bc \
    --trials 5 \
    --learning-rate 0.01 \
    --max-iterations 100

# 4. Validate trained weights
python3 scripts/graphbrew_experiment.py --validate-adaptive \
    --graphs all \
    --benchmarks pr bfs
```

---

## Slide 15: Results Flow

# End-to-End Example

```
Input: soc-LiveJournal1 (4.8M nodes, 68M edges)
       Social network with strong community structure

┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Extract Global Features                                 │
│   modularity = 0.71, degree_variance = 2.34                     │
│   hub_concentration = 0.45, avg_path_length = 5.2               │
│   diameter = 16, clustering_coeff = 0.28                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Find Best Type Match                                    │
│   → type_3 (social network cluster)                             │
│   → Load weights from type_3.json                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Leiden Detects 847 Communities                          │
│                                                                 │
│ Community 1 (450K nodes): dense, high clustering                │
│   → Features: density=0.08, hub_conc=0.38                       │
│   → Score: RABBITORDER=2.1, GraphBrewOrder=1.9, HubSort=1.5     │
│   → SELECT: RABBITORDER                                         │
│                                                                 │
│ Community 2 (120K nodes): sparse, hub-dominated                 │
│   → Features: density=0.002, hub_conc=0.72                      │
│   → Score: HubClusterDBG=2.4, RABBITORDER=1.6                   │
│   → SELECT: HubClusterDBG                                       │
│                                                                 │
│ Community 3 (50K nodes): small, uniform                         │
│   → SELECT: ORIGINAL (below threshold)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Apply Reordering & Merge                                │
│   → Apply RABBITORDER to Community 1                            │
│   → Apply HubClusterDBG to Community 2                          │
│   → Keep Community 3 as-is                                      │
│   → Combine in size order (largest first)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Result: PageRank speedup = 1.47x vs ORIGINAL                    │
│         (Best fixed algorithm: 1.35x)                           │
│         Adaptive advantage: +0.12x                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Slide 16: Key Innovations

# What Makes GraphBrew Different

### 1. **Per-Community Algorithm Selection**
Unlike traditional approaches that apply one algorithm globally,
GraphBrew selects the best algorithm for each community's characteristics.

### 2. **Online Learning with Incremental Updates**
Weights are updated immediately after each benchmark run,
not in batch mode. This enables:
- Faster convergence
- Continuous improvement
- Real-time adaptation

### 3. **Type-Based Specialization**
Auto-clustering groups similar graphs into types,
allowing specialized weights for each graph family.

### 4. **Extended Feature Set**
12 features capture both global graph structure and
local community characteristics, including topology metrics.

### 5. **Cache-Aware Training**
Optional cache simulation data helps tune weights for
specific memory hierarchy configurations.

---

## Slide 17: Performance Summary

# Expected Speedups

| Graph Type | Typical Speedup | Best Algorithm |
|------------|-----------------|----------------|
| Social Networks | 1.3-1.5x | LeidenOrder, RabbitOrder |
| Road Networks | 1.2-1.4x | RCM, CORDER |
| Web Graphs | 1.4-1.6x | HubClusterDBG, DBG |
| Power-Law | 1.3-1.5x | RabbitOrder, HubCluster |
| Uniform Random | 1.0-1.1x | ORIGINAL, DBG |

### Adaptive Advantage:
- **5-15% improvement** over best single algorithm
- **Handles mixed graphs** where no single algorithm wins
- **Robust across benchmarks** (PR, BFS, CC, SSSP, BC)

---

## Slide 18: Command Reference

# Key Commands

```bash
# Show available types and their characteristics
python3 scripts/graphbrew_experiment.py --show-types

# Train on specific benchmarks
python3 scripts/graphbrew_experiment.py --train-adaptive \
    --benchmarks pr bfs --graphs medium

# Validate current weights
python3 scripts/graphbrew_experiment.py --validate-adaptive \
    --graphs all --benchmarks pr

# Run with custom graph directory
python3 scripts/graphbrew_experiment.py --fill-weights \
    --graphs-dir /path/to/graphs

# Clean and restart training
python3 scripts/graphbrew_experiment.py --clean-all
python3 scripts/graphbrew_experiment.py --init-weights
python3 scripts/graphbrew_experiment.py --fill-weights --graphs all
```

---

## Slide 19: Architecture Summary

# System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GraphBrew System                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Python Training Pipeline                         │   │
│  │  (scripts/graphbrew_experiment.py)                           │   │
│  │                                                               │   │
│  │  • Graph discovery & download                                 │   │
│  │  • Benchmark orchestration                                    │   │
│  │  • Feature extraction from C++ output                         │   │
│  │  • Perceptron weight updates                                  │   │
│  │  • Type-based weight management                               │   │
│  └────────────────────────┬─────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              C++ Runtime Engine                               │   │
│  │  (bench/include/external/gapbs/builder.h)                     │   │
│  │                                                               │   │
│  │  • Leiden community detection (integrated)                    │   │
│  │  • Community feature computation                              │   │
│  │  • Perceptron score evaluation                                │   │
│  │  • Algorithm application per community                        │   │
│  │  • Weight file loading (JSON)                                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Weight Files                                     │   │
│  │  (results/weights/)                                           │   │
│  │                                                               │   │
│  │  • type_registry.json  (graph → type mapping)                 │   │
│  │  • type_0.json ... type_N.json (per-type weights)             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Slide 20: Conclusion

# Summary

### GraphBrew Capabilities:
✅ **Adaptive per-community algorithm selection**
✅ **ML-based perceptron with 12 graph features**
✅ **Online learning with incremental weight updates**
✅ **Type-based weight specialization**
✅ **16 reordering algorithms supported**
✅ **5 graph benchmarks (PR, BFS, CC, SSSP, BC)**
✅ **Cache simulation support**

### Key Results:
- **5-15% improvement** over best fixed algorithm
- **Automatic adaptation** to graph characteristics
- **Continuous learning** from benchmark results

### Future Work:
- GPU-accelerated community detection
- Larger training corpus
- Neural network score function
- Multi-level recursive adaptive ordering

---

*GraphBrew - Making Graph Processing Cache-Efficient Through Intelligent Reordering*
