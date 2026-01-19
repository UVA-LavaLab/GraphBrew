# GraphBrewOrder: Per-Community Reordering

GraphBrewOrder (algorithm 13) is the core reordering algorithm that applies **different optimizations to different communities**. This page explains how it works and when to use it.

## Overview

Unlike traditional reordering that treats the entire graph uniformly, GraphBrewOrder:

1. **Detects communities** using Leiden
2. **Analyzes each community's structure**
3. **Applies hub-based reordering** within communities
4. **Preserves community locality** in the final ordering

```
Input Graph → Leiden → Communities → Per-Community Hub Order → Output Graph
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
│ [hubs][non-hubs ] [hubs][non-hubs ] [hubs][non-hubs ] │
│ Both community AND hub locality!                        │
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
struct CommunityStats {
    int32_t num_nodes;
    int64_t num_edges;
    double density;
    double avg_degree;
    double degree_variance;
    double hub_concentration;  // edges to top 10%
};
```

### Step 3: Per-Community Ordering

Within each community, vertices are ordered by degree (hub priority):

```
Community 0 (before):
  Vertices: [45, 12, 89, 3, 67, 23]
  Degrees:  [5,  120, 8, 95, 12, 45]

Community 0 (after hub sort):
  Vertices: [12, 3, 23, 67, 89, 45]
  Degrees:  [120, 95, 45, 12, 8, 5]
  
  Hubs first, then non-hubs
```

### Step 4: Combine Communities

Communities are concatenated in order of size (largest first):

```
Final ordering:
[Community 2 (5000 nodes)] [Community 0 (3000 nodes)] [Community 1 (1500 nodes)]
[hub-sorted internally  ] [hub-sorted internally  ] [hub-sorted internally  ]
```

---

## How to Use

### Basic Usage

```bash
# Use GraphBrewOrder (algorithm 13)
./bench/bin/pr -f graph.el -s -o 13 -n 5
```

### With Community Output

```bash
# Verbose output shows community breakdown
./bench/bin/pr -f graph.el -s -o 13 -n 1 2>&1 | head -30
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

### vs LeidenOrder (algorithm 12)

| Aspect | LeidenOrder | GraphBrewOrder |
|--------|-------------|----------------|
| Community detection | ✅ Leiden | ✅ Leiden |
| Internal ordering | Original | Hub-sorted |
| Hub optimization | ❌ None | ✅ Yes |
| Best for | Basic community grouping | Hub-heavy communities |

### vs LeidenHybrid (algorithm 20)

| Aspect | LeidenHybrid | GraphBrewOrder |
|--------|--------------|----------------|
| Community detection | ✅ Leiden | ✅ Leiden |
| Internal ordering | DFS traversal | Hub sort |
| Complexity | Higher | Lower |
| Best for | Complex graphs | Simple hub-based |

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
Communities grouped, hubs first within each:

Memory: [v12][v234][v167][v892][v45][v8923][v5601][v4501]...
        └──Community 0──────┘└──Community 1──┘└──Community 2─┘
        └hubs┘└─non-hubs───┘ └hub┘└─non-hub─┘ └hub──┘└non-hub┘

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

### Hub Threshold

Within each community, "hubs" are defined as:

```cpp
// Top 10% by degree within the community
int hub_threshold = community_degrees[community_size * 0.9];
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

### Minimum Community Size

Small communities don't benefit from hub sorting:

```cpp
// Communities smaller than threshold keep original order
const int MIN_COMMUNITY_FOR_ORDERING = 50;
```

### Hub Percentile

Adjust what fraction of vertices are considered "hubs":

```cpp
// Default: top 10%
const double HUB_PERCENTILE = 0.10;

// More aggressive (top 5%)
const double HUB_PERCENTILE = 0.05;
```

---

## Combining with Other Algorithms

### GraphBrewOrder + DBG

Apply Degree-Based Grouping after GraphBrewOrder:

```bash
# First apply GraphBrewOrder, then DBG
./bench/bin/pr -f graph.el -s -o 13 -n 1  # GraphBrewOrder
# Manual DBG pass would need code modification
```

### AdaptiveOrder Uses GraphBrewOrder

AdaptiveOrder (algorithm 15) can select GraphBrewOrder for appropriate communities:

```json
// In perceptron_weights.json
{
  "GraphBrewOrder": {
    "bias": 0.7,
    "w_modularity": 0.2,
    "w_hub_concentration": 0.25
  }
}
```

---

## Benchmarking GraphBrewOrder

### Single Benchmark
```bash
./bench/bin/pr -f graph.el -s -o 13 -n 10
```

### Compare with Alternatives
```bash
# Test different algorithms
for algo in 0 4 7 12 13 20; do
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
Algorithm 12 (LeidenOrder):  Average: 0.0398 seconds
Algorithm 13 (GraphBrewOrder): Average: 0.0385 seconds ← Good
Algorithm 20 (LeidenHybrid): Average: 0.0371 seconds ← Better
```

---

## Troubleshooting

### "Too many small communities"

If Leiden creates too many tiny communities:

```bash
# Check community distribution
./bench/bin/pr -f graph.el -s -o 13 -n 1 2>&1 | grep -A 10 "Community Size"
```

If most communities have < 10 vertices, the graph may have weak community structure.

### "Worse than HUBCLUSTER"

For graphs without clear communities, global hub ordering may be better:

```bash
# Compare
./bench/bin/pr -f graph.el -s -o 7 -n 5   # HUBCLUSTERDBG
./bench/bin/pr -f graph.el -s -o 13 -n 5  # GraphBrewOrder
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
