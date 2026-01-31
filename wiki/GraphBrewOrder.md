# GraphBrewOrder: Per-Community Reordering

GraphBrewOrder (algorithm 12) is the core reordering algorithm that applies **different optimizations to different communities**. This page explains how it works and when to use it.

## Overview

Unlike traditional reordering that treats the entire graph uniformly, GraphBrewOrder:

1. **Detects communities** using Leiden
2. **Analyzes each community's structure**
3. **Applies configurable reordering** within communities (default: RabbitOrder)
4. **Preserves community locality** in the final ordering

```
Input Graph → Leiden → Communities → Per-Community Reorder → Output Graph
```

## Architecture Flow Diagram

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
| Comm0 | | Comm1 | | CommN |
| 15234 | | 8912  | | 500   |
| nodes | | nodes | | nodes |
+---+---+ +---+---+ +---+---+
    |         |         |
    v         v         v
+-------+ +-------+ +-------+
|REORDER| |REORDER| |REORDER|
| (per- | | (per- | | (per- |
| comm) | | comm) | | comm) |
+---+---+ +---+---+ +---+---+
    |         |         |
    +----+----+----+----+
         |
         v
+------------------+
| SIZE-SORTED MERGE|
| [Comm0][Comm1]...|
+------------------+
         |
         v
+------------------+
|  OUTPUT GRAPH    |
| (reordered IDs)  |
+------------------+
```

---

## Why Per-Community?

### The Problem with Global Reordering

Global algorithms like HUBCLUSTER or RCM optimize for one metric across the entire graph:

```
Global HUBCLUSTER:
┌─────────────────────────────────────────────────────────┐
│ [All Hubs] ──────── [All Non-Hubs]                      │
│ Good hub locality, but communities are scattered        │
└─────────────────────────────────────────────────────────┘
```

This can **break community locality** - vertices from different communities get interleaved.

### The GraphBrewOrder Solution

GraphBrewOrder keeps communities together while optimizing within each:

```
GraphBrewOrder:
┌─────────────────────────────────────────────────────────┐
│ [Community 0     ] [Community 1     ] [Community 2     ]│
│ [  reordered    ] [  reordered    ] [  reordered    ] │
│ Both community AND cache locality!                      │
└─────────────────────────────────────────────────────────┘
```

---

## Algorithm Steps

### Step 1: Community Detection

```cpp
// Leiden community detection
auto communities = leidenCommunityDetection(graph);

// Result: each vertex mapped to a community ID
// vertex 0 → community 3
// vertex 1 → community 0
// vertex 2 → community 3
// ...
```

### Step 2: Community Analysis

For each community, compute:

```cpp
struct CommunityFeatures {
    size_t num_nodes;
    size_t num_edges;
    double internal_density;     // edges / possible_edges
    double avg_degree;
    double degree_variance;      // normalized variance in degrees
    double hub_concentration;    // fraction of edges from top 10% nodes
    double modularity;           // community/subgraph modularity
    double clustering_coeff;     // local clustering coefficient (sampled)
    double avg_path_length;      // estimated average path length
    double diameter_estimate;    // estimated graph diameter
    double community_count;      // number of sub-communities
    double reorder_time;         // estimated reorder time (if known)
};
```

### Step 3: Per-Community Ordering

Within each community, vertices are reordered using a configurable algorithm (default: RabbitOrder):

```
Community 0 (before):
  Vertices: [45, 12, 89, 3, 67, 23]
  
Community 0 (after per-community reordering):
  Vertices: [12, 3, 23, 67, 89, 45]
  
  RabbitOrder optimizes cache locality within the community subgraph
```

The algorithm used for per-community ordering can be configured via the format string (see Usage section).

### Step 4: Combine Communities

Communities are concatenated in order of size (largest first):

```
Final ordering:
[Community 2 (5000 nodes)] [Community 0 (3000 nodes)] [Community 1 (1500 nodes)]
[reordered internally   ] [reordered internally   ] [reordered internally   ]
```

---

## How to Use

### Basic Usage

```bash
# Use GraphBrewOrder (algorithm 12)
./bench/bin/pr -f graph.el -s -o 12 -n 5
```

### With Options

```bash
# Format: -o 12:freq:algo:resolution:maxIterations:maxPasses
# freq: frequency threshold for small communities (default: 10)
# algo: per-community ordering algorithm ID (default: 8 = RabbitOrder)
# resolution: Leiden resolution parameter (default: auto based on density)
# maxIterations: maximum Leiden iterations (default: 30)
# maxPasses: maximum Leiden passes (default: 30)

# Use HubClusterDBG (7) for per-community ordering with resolution 1.0
./bench/bin/pr -f graph.el -s -o 12:10:7:1.0

# Full example with all parameters (explicit resolution)
./bench/bin/pr -f graph.el -s -o 12:10:8:1.0:30:30
```

### With Community Output

```bash
# Verbose output shows community breakdown
./bench/bin/pr -f graph.el -s -o 12 -n 1 2>&1 | head -30
```

Output:
```
=== GraphBrew Per-Community Reordering ===
Communities detected: 847
Total reorder time: 0.234 seconds

Community Size Distribution:
  < 100 nodes:   812 communities
  100-1000:      28 communities  
  1000-10000:    6 communities
  > 10000:       1 community

Largest communities:
  Comm 0: 15,234 nodes, 189,456 edges (hub_conc: 0.58)
  Comm 1: 8,912 nodes, 98,234 edges (hub_conc: 0.42)
  ...
```

---

## Comparison with Other Algorithms

### vs HUBCLUSTER (algorithm 4)

| Aspect | HUBCLUSTER | GraphBrewOrder |
|--------|------------|----------------|
| Hub locality | ✅ Excellent | ✅ Good (per-community) |
| Community locality | ❌ Poor | ✅ Excellent |
| Scalability | ✅ Fast | ⚡ Moderate |
| Best for | Uniform hub graphs | Modular graphs |

### vs LeidenOrder (algorithm 15)

| Aspect | LeidenOrder | GraphBrewOrder |
|--------|-------------|----------------|
| Community detection | ✅ Leiden | ✅ Leiden |
| Internal ordering | Original | Per-community (RabbitOrder) |
| Cache optimization | ❌ None | ✅ Yes |
| Best for | Basic community grouping | Better cache locality |

### vs LeidenCSR (algorithm 17)

| Aspect | LeidenCSR | GraphBrewOrder |
|--------|-----------|----------------|
| Community detection | ✅ Leiden | ✅ Leiden |
| Internal ordering | Configurable (gve/gveopt/gverabbit/dfs/bfs/hubsort/fast/modularity) | Configurable (default: RabbitOrder) |
| Speed | Very fast (CSR-native) | Moderate |
| Best for | Large graphs, tunable quality | Modular graphs |

---

## Visual Example

### Original Graph
```
Vertex IDs scattered, poor locality:

Memory: [v892][v12][v5601][v234][v8923][v45][v167][v4501]...
         └─C0─┘└C1┘└─C2──┘└C0─┘└─C1──┘└C2┘└─C0──┘└C2───┘

Cache lines load vertices from multiple communities
= Many cache misses during graph traversal
```

### After GraphBrewOrder
```
Communities grouped, optimized within each:

Memory: [v12][v234][v167][v892][v45][v8923][v5601][v4501]...
        └──Community 0──────┘└──Community 1──┘└──Community 2─┘
        └── optimized ─────┘ └── optimized ─┘ └─ optimized ─┘

Sequential access within communities
= Fewer cache misses
```

---

## Performance Characteristics

### When GraphBrewOrder Excels

✅ **Modular graphs** with clear community structure
✅ **Social networks** (communities = friend groups)
✅ **Web graphs** (communities = websites/domains)
✅ **Citation networks** (communities = research areas)

### When to Use Alternatives

❌ **Road networks** - low modularity, use RCM instead
❌ **Random graphs** - no community structure, use HUBCLUSTER
❌ **Very small graphs** - overhead not worth it, use ORIGINAL

### Overhead

```
Leiden detection:  ~10-15% of total time
Hub sorting:       ~2-5% of total time
Memory:            O(n) for community mapping
```

For graphs > 10K vertices, the overhead is usually recovered through better cache performance.

---

## Implementation Details

### C++ Code Architecture

GraphBrewOrder's implementation is organized in modular header files in `bench/include/gapbs/reorder/`:

| File | Purpose |
|------|---------|
| `reorder_graphbrew.h` | `GraphBrewConfig` struct, `GraphBrewClusterVariant` enum |
| `reorder_types.h` | Base types, perceptron model, feature computation |

**GraphBrewClusterVariant Enum:**

```cpp
// bench/include/gapbs/reorder/reorder_graphbrew.h
enum class GraphBrewClusterVariant {
    LEIDEN,      // Original Leiden library (igraph)
    GVE,         // GVE-Leiden CSR-native
    GVEOPT,      // Cache-optimized GVE
    GVEFAST,     // Single-pass GVE
    GVEOPTFAST,  // Cache-optimized single-pass
    RABBIT,      // RabbitOrder-based
    HUBCLUSTER   // Hub-clustering based
};
```

**GraphBrewConfig Struct:**

```cpp
// bench/include/gapbs/reorder/reorder_graphbrew.h
struct GraphBrewConfig {
    GraphBrewClusterVariant variant = GraphBrewClusterVariant::LEIDEN;
    int frequency = 10;           // Hub frequency threshold
    int intra_algo = 8;           // Algorithm ID for within-community reordering
    double resolution = -1.0;     // Leiden resolution (-1 = auto)
    int maxIterations = 30;       // Max Leiden iterations
    int maxPasses = 30;           // Max Leiden passes

    // Parse from command-line options string
    static GraphBrewConfig FromOptions(const std::string& options);
    
    // Convert to internal reordering options
    ReorderingOptions toInternalOptions() const;
    
    // Print configuration for debugging
    void print() const;
};

// Usage in builder.h:
GraphBrewConfig config = GraphBrewConfig::FromOptions("gve:10:8:1.0:30:30");
// → variant=GVE, frequency=10, intra_algo=8, resolution=1.0, etc.
```

**Key Functions in builder.h:**

```cpp
// Unified entry point for GraphBrewOrder
void GenerateGraphBrewMappingUnified(
    const CSRGraph& g, pvector<NodeID_>& new_ids,
    const ReorderingOptions& opts);
```

### Community Ordering Strategy

Communities are processed in this order:

```cpp
// Sort communities by size (largest first)
// Rationale: largest communities benefit most from optimization
sort(communities.begin(), communities.end(), 
     [](auto& a, auto& b) { return a.size > b.size; });
```

### Per-Community Reordering

Each community subgraph is reordered using the configurable algorithm:

```cpp
// Default: RabbitOrder (algorithm 8 with csr variant)
// RabbitOrder has variants: csr (default), boost
// Can be changed via format string: -o 12:freq:algo:resolution
ReorderingAlgo algo = getReorderingAlgo(reorderingOptions[1].c_str());  // default: "8"

// Apply the selected algorithm to each community subgraph
GenerateMappingLocalEdgelist(g, edge_list, new_ids_sub, algo, ...);
```

### Vertex Relabeling

```cpp
// New vertex ID assignment
new_id = community_start_offset + position_within_community

// Example:
// Community 0 starts at offset 0, has 1000 vertices
// Community 1 starts at offset 1000, has 500 vertices
// Vertex 45 in Community 1 (position 23) gets ID: 1000 + 23 = 1023
```

---

## GraphBrewOrder Variants

GraphBrewOrder supports multiple community detection backends via variant selection:

| Variant | Description |
|---------|-------------|
| `leiden` | **Default.** Original Leiden library (igraph-based) |
| `gve` | GVE-Leiden CSR-native implementation |
| `gveopt` | Cache-optimized GVE with prefetching |
| `gvefast` | Single-pass GVE (faster, less refinement) |
| `gveoptfast` | Cache-optimized single-pass GVE |
| `rabbit` | RabbitOrder-based community detection |
| `hubcluster` | Hub-clustering based approach |

### Usage in Python scripts

```bash
# Test specific GraphBrewOrder variants
python3 scripts/graphbrew_experiment.py --phase benchmark \
  --graph wiki-topcats \
  --algo-list GraphBrewOrder \
  --all-variants \
  --graphbrew-variants leiden gve gveopt \
  --benchmarks pr --trials 3
```

### Usage with C++ binaries

```bash
# Format: -o 12:variant
./bench/bin/pr -f graph.el -s -o 12:leiden -n 5   # Original Leiden
./bench/bin/pr -f graph.el -s -o 12:gve -n 5      # GVE-Leiden
./bench/bin/pr -f graph.el -s -o 12:gveopt -n 5   # Cache-optimized GVE
```

---

## Configuring GraphBrewOrder

### Dynamic Community Size Thresholds

The implementation uses **dynamic** thresholds based on graph statistics:

```cpp
// Dynamic threshold formula:
// min_size = max(ABSOLUTE_MIN, min(avg_community_size / FACTOR, sqrt(N)))
// Capped at MAX_THRESHOLD (2000)

size_t ComputeDynamicMinCommunitySize(size_t num_nodes, 
                                       size_t num_communities,
                                       size_t avg_community_size) {
    const size_t ABSOLUTE_MIN = 50;
    const size_t MAX_THRESHOLD = 2000;
    const size_t FACTOR = 4;
    
    size_t dynamic_threshold = std::max(ABSOLUTE_MIN,
        std::min(avg_community_size / FACTOR, 
                 static_cast<size_t>(std::sqrt(num_nodes))));
    
    return std::min(dynamic_threshold, MAX_THRESHOLD);
}
```

This ensures thresholds scale appropriately across different graph sizes.

### Local Reorder Threshold

Communities must exceed this size to apply internal reordering:

```cpp
// Local reorder threshold = 2 × min_community_size (capped at 5000)
size_t ComputeDynamicLocalReorderThreshold(size_t num_nodes,
                                            size_t num_communities,
                                            size_t avg_community_size) {
    size_t min_size = ComputeDynamicMinCommunitySize(...);
    return std::min(min_size * 2, static_cast<size_t>(5000));
}
```

### Grouped Small Communities

Small communities (below the threshold) are grouped together and processed as a single "mega community" using feature-based algorithm selection:

```cpp
// Small communities grouped together
// Features computed for the entire group
// Algorithm selected via perceptron (AdaptiveOrder) or heuristic (GraphBrew)
```

---

## Combining with Other Algorithms

### GraphBrewOrder + DBG

Apply Degree-Based Grouping after GraphBrewOrder:

```bash
# First apply GraphBrewOrder, then DBG
./bench/bin/pr -f graph.el -s -o 12 -n 1  # GraphBrewOrder
# Manual DBG pass would need code modification
```

### AdaptiveOrder Uses GraphBrewOrder

AdaptiveOrder (14) can select GraphBrewOrder for appropriate communities. Per-type perceptron weights are stored in:

```bash
scripts/weights/active/type_N.json    # Per-cluster weights
scripts/weights/active/type_registry.json  # Graph → type mapping
```

GraphBrewOrder default weights (from builder.h):

```cpp
// GraphBrewOrder: Leiden + per-community RabbitOrder
{GraphBrewOrder, {
    .bias = 0.6,
    .w_modularity = -0.25,    // better on low-mod graphs
    .w_log_nodes = 0.1,       // scales well
    .w_log_edges = 0.08,
    .w_density = -0.3,
    .w_avg_degree = 0.0,
    .w_degree_variance = 0.2,
    .w_hub_concentration = 0.1,
    .w_reorder_time = -0.5    // highest reorder cost
}}
```

---

## Benchmarking GraphBrewOrder

### Single Benchmark
```bash
./bench/bin/pr -f graph.el -s -o 12 -n 10
```

### Compare with Alternatives
```bash
# Test different algorithms
for algo in 0 4 7 12 15 17; do
    echo "Algorithm $algo:"
    ./bench/bin/pr -f graph.el -s -o $algo -n 5 2>&1 | grep "Average"
done
```

### Expected Results

For social networks:
```
Algorithm 0 (ORIGINAL):      Average: 0.0523 seconds
Algorithm 4 (HUBCLUSTER):    Average: 0.0445 seconds
Algorithm 7 (HUBCLUSTERDBG): Average: 0.0412 seconds
Algorithm 12 (GraphBrewOrder): Average: 0.0385 seconds ← Good
Algorithm 15 (LeidenOrder):  Average: 0.0398 seconds
Algorithm 17 (LeidenCSR):    Average: 0.0371 seconds ← Better
```

---

## Troubleshooting

### "Too many small communities"

If Leiden creates too many tiny communities:

```bash
# Check community distribution
./bench/bin/pr -f graph.el -s -o 12 -n 1 2>&1 | grep -A 10 "Community Size"
```

If most communities have < 10 vertices, the graph may have weak community structure.

### "Worse than HUBCLUSTER"

For graphs without clear communities, global hub ordering may be better:

```bash
# Compare
./bench/bin/pr -f graph.el -s -o 7 -n 5   # HUBCLUSTERDBG
./bench/bin/pr -f graph.el -s -o 12 -n 5  # GraphBrewOrder
```

If HUBCLUSTERDBG is faster, the graph has weak community structure.

### "Memory issues"

GraphBrewOrder needs memory for:
- Community mapping: O(n) integers
- Temporary degree arrays: O(n) integers

For very large graphs:
```bash
# Check memory before running
free -h

# GraphBrewOrder needs ~16 bytes per vertex
# 1B vertices ≈ 16GB additional memory
```

---

## Next Steps

- [[AdaptiveOrder-ML]] - Automatic algorithm selection per community
- [[Reordering-Algorithms]] - All available algorithms
- [[Running-Benchmarks]] - Benchmark commands

---

[← Back to Home](Home) | [Reordering Algorithms →](Reordering-Algorithms)
