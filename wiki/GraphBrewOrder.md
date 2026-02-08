# GraphBrewOrder: Per-Community Reordering

GraphBrewOrder (algorithm 12) is the core reordering algorithm that applies **different optimizations to different communities**. This page explains how it works and when to use it.

## Overview

Unlike traditional reordering that treats the entire graph uniformly, GraphBrewOrder:

1. **Detects communities** using Leiden
2. **Analyzes each community's structure**
3. **Applies configurable reordering** within communities (default: RabbitOrder)
4. **Preserves community locality** in the final ordering

```
Input Graph → Leiden → Communities → Per-Community Reorder → Size-Sorted Merge → Output
```

**Why per-community?** Global algorithms (HUBCLUSTER, RCM) break community locality by interleaving vertices from different communities. GraphBrewOrder keeps communities contiguous while optimizing cache locality within each.

---

## Algorithm Steps

1. **Community Detection**: Leiden maps each vertex to a community ID
2. **Community Analysis**: Compute features per community (density, degree variance, hub concentration, etc.) — see `CommunityFeatures` struct in [[Code-Architecture]]
3. **Per-Community Ordering**: Reorder within each community using configurable algorithm (default: RabbitOrder)
4. **Size-Sorted Merge**: Concatenate communities largest-first → final vertex IDs

---

## How to Use

### Basic Usage

```bash
# Use GraphBrewOrder (algorithm 12) with auto-resolution
./bench/bin/pr -f graph.el -s -o 12 -n 5
```

### New Format (Recommended)

```bash
# Format: -o 12:cluster_variant:final_algo:resolution:levels
# cluster_variant: leiden (default), gve, gveopt, rabbit, hubcluster, gvefast, gveoptfast
# final_algo: per-community ordering algorithm ID (default: 8 = RabbitOrder)
# resolution: Leiden resolution parameter (default: dynamic for best PR)
# levels: recursion depth (default: 1)

# Examples:
./bench/bin/pr -f graph.el -s -o 12:gveopt -n 5           # GVE-Leiden optimized, auto-resolution
./bench/bin/pr -f graph.el -s -o 12:rabbit -n 5           # Coarse communities (resolution=0.5)
./bench/bin/pr -f graph.el -s -o 12:gveopt:6 -n 5         # Use HubSortDBG as final algorithm
./bench/bin/pr -f graph.el -s -o 12:gveopt:8:0.75 -n 5    # Custom resolution 0.75
```

### Old Format (Backward Compatible)

```bash
# Format: -o 12:freq:algo:resolution:maxIterations:maxPasses
# freq: frequency threshold for small communities (default: 10)
# algo: per-community ordering algorithm ID (default: 8 = RabbitOrder)
# resolution: Leiden resolution parameter (default: dynamic)

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

| Aspect | HUBCLUSTER (4) | LeidenOrder (15) | LeidenCSR (17) | GraphBrewOrder (12) |
|--------|---------------|-----------------|---------------|--------------------|
| Community locality | ❌ Poor | ✅ Good | ✅ Good | ✅ Excellent |
| Internal ordering | Hub sort | Original | Configurable | Per-community (RabbitOrder) |
| Speed | Fast | Moderate | Very fast | Moderate |
| Best for | Uniform hub graphs | Basic grouping | Large graphs | Modular graphs |

---

## When to Use

✅ **Modular graphs**: social networks, web graphs, citation networks (modularity > 0.3)
❌ **Not ideal for**: road networks (low modularity), random graphs, very small graphs (< 10K vertices)

**Overhead**: Leiden ~10-15% + hub sorting ~2-5% of total time, O(n) memory. Usually recovered through better cache performance on graphs > 10K vertices.

---

## Implementation Details

### C++ Code Architecture

GraphBrewOrder's implementation is organized in modular header files in `bench/include/graphbrew/reorder/`:

| File | Purpose |
|------|---------|
| `reorder_graphbrew.h` | `GraphBrewConfig` struct, `GraphBrewClusterVariant` enum |
| `reorder_types.h` | Base types, perceptron model, feature computation |

**GraphBrewClusterVariant Enum:**

```cpp
// bench/include/graphbrew/reorder/reorder_graphbrew.h
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
// bench/include/graphbrew/reorder/reorder_graphbrew.h
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

### Edge Case Handling

GraphBrewOrder includes guards for graphs with extreme community structure (e.g., Kronecker graphs):

```cpp
// In ReorderCommunitySubgraphStandalone (reorder.h):
// If community has no internal edges (all edges go to other communities),
// skip reordering and assign nodes in original order
if (sub_edges.empty()) {
    for (NodeID_ node : nodes) {
        new_ids[node] = current_id++;
    }
    return;
}
```

This prevents division-by-zero errors when:
- Leiden creates communities where nodes only connect to OTHER communities
- The induced subgraph has 0 edges
- Hub-based algorithms compute `avgDegree = num_edges / num_nodes`

**Graphs that benefit from this handling:**
- Kronecker graphs (kron_g500-logn*)
- Synthetic power-law graphs with extreme degree distributions
- Graphs with highly disconnected community structure

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

## Variants

| Variant | Description | Final Algorithm |
|---------|-------------|----------------|
| `leiden` | **Default.** GVE-Leiden optimized, auto-resolution | RabbitOrder |
| `gve` / `gveopt` | GVE-Leiden CSR-native / cache-optimized | RabbitOrder |
| `gvefast` / `gveoptfast` | Single-pass variants | HubSortDBG |
| `rabbit` | Resolution=0.5 (coarser communities) | RabbitOrder |
| `hubcluster` | Hub-degree based clustering | RabbitOrder |

```bash
# Format: -o 12:variant:final_algo:resolution:levels
./bench/bin/pr -f graph.el -s -o 12:gveopt -n 5           # Cache-optimized GVE
./bench/bin/pr -f graph.el -s -o 12:gveopt:6 -n 5         # Custom final algo (HubSortDBG)
./bench/bin/pr -f graph.el -s -o 12:gveopt:8:0.75 -n 5    # Custom resolution
```

See [[Command-Line-Reference]] for full variant list and [[Python-Scripts]] for experiment integration.

---

## Configuration

- **Dynamic thresholds**: `min_size = max(50, min(avg_comm_size/4, √N))`, capped at 2000
- **Local reorder threshold**: `2 × min_size`, capped at 5000
- **Small communities**: Grouped into a "mega community" for perceptron-based algorithm selection
- **AdaptiveOrder integration**: Can select GraphBrewOrder per-community — see [[AdaptiveOrder-ML]]

---

## Benchmarking

```bash
./bench/bin/pr -f graph.el -s -o 12 -n 10  # Single benchmark

# Compare with alternatives
for algo in 0 4 7 12 15 17; do
    echo "Algorithm $algo:"
    ./bench/bin/pr -f graph.el -s -o $algo -n 5 2>&1 | grep "Average"
done
```

Typical social network result: GraphBrewOrder ~1.36x faster than ORIGINAL, LeidenCSR slightly faster still.

See [[Troubleshooting]] for common issues (small communities, worse than HUBCLUSTER, memory).

---

## Next Steps

- [[AdaptiveOrder-ML]] - Automatic algorithm selection per community
- [[Reordering-Algorithms]] - All available algorithms
- [[Running-Benchmarks]] - Benchmark commands

---

[← Back to Home](Home) | [Reordering Algorithms →](Reordering-Algorithms)
