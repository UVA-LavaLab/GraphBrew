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
| Internal ordering | Configurable (dfs/bfs/hubsort/fast) | Configurable (default: RabbitOrder) |
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
// Default: RabbitOrder (algorithm 8)
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

## Configuring GraphBrewOrder

### Community Size Thresholds

The implementation uses several size thresholds for optimization:

```cpp
// Communities smaller than this use simplified processing
const size_t MIN_COMMUNITY_SIZE = 200;

// Minimum community size for local reordering
const size_t MIN_COMMUNITY_FOR_LOCAL_REORDER = 500;

// Large communities get full parallelism during reordering
const size_t LARGE_COMMUNITY_THRESHOLD = 500000;  // 500K edges
```

### Frequency Threshold

Communities with fewer vertices than the frequency threshold are merged:

```cpp
// Default frequency threshold (configurable via format string)
size_t frequency_threshold = 10;
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
