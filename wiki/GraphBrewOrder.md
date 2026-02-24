# GraphBrewOrder: Per-Community Reordering (GraphBrew-Powered)

GraphBrewOrder (algorithm 12) is the core reordering algorithm that applies **different optimizations to different communities**. As of 2026, GraphBrewOrder is powered by the **GraphBrew pipeline**, using its modular Leiden community detection and then applying per-community reordering.

## Overview

Unlike traditional reordering that treats the entire graph uniformly, GraphBrewOrder:

1. **Detects communities** using GraphBrew's Leiden pipeline (configurable aggregation)
2. **Classifies communities** into small/large using dynamic thresholds
3. **Merges small communities** and applies heuristic algorithm selection
4. **Applies configurable reordering** within large communities (default: RabbitOrder)
5. **Preserves community locality** in the final ordering

```
Input Graph → GraphBrew Leiden → Communities → Classify → Per-Community Reorder → Merge → Output
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
# Format: -o 12:cluster_variant:final_algo:resolution
# cluster_variant: leiden (default), rabbit, hubcluster
# final_algo: per-community ordering algorithm ID (default: 8 = RabbitOrder)
# resolution: Leiden resolution parameter (default: auto-computed from graph)

# Examples:
./bench/bin/pr -f graph.el -s -o 12:leiden -n 5            # Leiden detection, RabbitOrder final
./bench/bin/pr -f graph.el -s -o 12:rabbit -n 5           # GraphBrew RabbitOrder single-pass
./bench/bin/pr -f graph.el -s -o 12:leiden:6 -n 5          # Use HubSortDBG as final algorithm
./bench/bin/pr -f graph.el -s -o 12:leiden:8:0.75 -n 5     # Custom resolution 0.75
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

The output shows detected communities, reorder time, community size distribution, and per-community hub concentration. Run on your graph to see actual values.

---

## Comparison with Other Algorithms

| Aspect | HUBCLUSTER (4) | LeidenOrder (15) | GraphBrewOrder (12) |
|--------|---------------|-----------------|--------------------|
| Community locality | ❌ Poor | ✅ Good | ✅ Excellent |
| Internal ordering | Hub sort | Original | Per-community (RabbitOrder) |
| Speed | Fast | Moderate | Moderate |
| Best for | Uniform hub graphs | Basic grouping | Modular graphs |

---

## When to Use

✅ **Modular graphs**: social networks, web graphs, citation networks (modularity > 0.3)
❌ **Not ideal for**: road networks (low modularity), random graphs, very small graphs (< 10K vertices)

**Overhead**: Leiden community detection + hub sorting add some overhead (O(n) memory). Run the pipeline on your target graphs to measure actual overhead and cache performance trade-offs.

---

## Implementation Details

### Architecture: GraphBrew-Powered Pipeline

GraphBrewOrder uses GraphBrew's modular Leiden community detection, then applies per-community reordering:

| Component | File | Purpose |
|-----------|------|---------|
| CLI → GraphBrewConfig | `builder.h` | `ParseGraphBrewConfig()` |
| Main entry point | `builder.h` | `GenerateGraphBrewMappingUnified()` |
| Community detection | `reorder_graphbrew.h` | `graphbrew::runGraphBrew()` (Leiden pipeline) |
| Per-community dispatch | `reorder.h` | `ReorderCommunitySubgraphStandalone()` |
| Small community heuristic | `reorder_types.h` | `SelectAlgorithmForSmallGroup()` |
| Config types | `reorder_graphbrew.h` | `GraphBrewConfig` (supersedes old `GraphBrewConfig`) |

### Cluster Variant → GraphBrew Configuration Mapping

```cpp
// In builder.h: ParseGraphBrewConfig()
"leiden"     → GraphBrew GVE-CSR aggregation, TOTAL_EDGES M, refinement depth 0
"rabbit"     → GraphBrew RabbitOrder algorithm, resolution=0.5
"hubcluster" → GraphBrew Leiden + HUB_CLUSTER ordering (native, no external dispatch)
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

Each large community subgraph is reordered using the configured final algorithm:

```cpp
// Default: RabbitOrder (algorithm 8)
// Can be any algorithm 0-11 via format: -o 12:variant:algo_id
ReorderCommunitySubgraphStandalone(g, nodes, node_set, finalAlgo, useOutdeg, new_ids, current_id);
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

| Variant | Description | GraphBrew Configuration |
|---------|-------------|-------------------|
| `leiden` | **Default.** GraphBrew Leiden-CSR with GVE aggregation | GVE-CSR, TOTAL_EDGES, refine depth 0 |
| `rabbit` | GraphBrew RabbitOrder single-pass | RABBIT_ORDER algorithm, resolution=0.5 |
| `hubcluster` | GraphBrew Leiden + hub-cluster ordering | HUB_CLUSTER ordering (native GraphBrew) |

```bash
# Format: -o 12:variant:final_algo:resolution
./bench/bin/pr -f graph.el -s -o 12:leiden -n 5            # Leiden detection + RabbitOrder
./bench/bin/pr -f graph.el -s -o 12:leiden:6 -n 5          # Leiden + HubSortDBG
./bench/bin/pr -f graph.el -s -o 12:leiden:8:0.75 -n 5     # Custom resolution
```

See [[Command-Line-Reference]] for full variant list and [[Python-Scripts]] for experiment integration.

---

## Multi-Layer Configuration

GraphBrewOrder's `-o 12:...` string is parsed as a multi-layer pipeline. Each layer is an independent dimension that can be combined freely.

### Layer 0: Preset (one-of)
Selects the algorithm skeleton and defaults.

| Preset | Description |
|--------|-------------|
| `leiden` | GVE-CSR Leiden + per-community RabbitOrder **(default)** |
| `rabbit` | Full RabbitOrder pipeline (single-pass, no Leiden) |
| `hubcluster` | Leiden + hub-cluster native ordering |

### Layer 1: Ordering Strategy (one-of)
How vertices are permuted within/across communities.

| Strategy | Description |
|----------|-------------|
| `hrab` | Hybrid Leiden+Rabbit BFS **(best general locality)** |
| `dfs` | DFS traversal of community dendrogram |
| `bfs` | BFS traversal of community dendrogram |
| `conn` | Connectivity BFS within communities |
| `dbg` | Degree-Based Grouping within communities |
| `corder` | Hot/cold partitioning within communities |
| `dbg-global` | DBG across all vertices (post-clustering) |
| `corder-global` | Corder across all vertices |
| `hierarchical` | Multi-level sort by all Leiden passes |
| `hcache` | Cache-aware hierarchical ordering |
| `tqr` | Tile-quantized graph + RabbitOrder |

See `GRAPHBREW_LAYERS["ordering"]` in `scripts/lib/utils.py` for the complete list.

### Layer 2: Aggregation Strategy (one-of)
How the community super-graph is built.

| Strategy | Description |
|----------|-------------|
| `gvecsr` | GVE-style explicit super-graph **(best quality)** |
| `leiden` | Standard Leiden CSR aggregation |
| `streaming` | RabbitOrder-style lazy incremental merge (fast) |
| `hybrid` | Lazy for early passes, CSR for final |

### Layer 3: Feature Flags (additive)
Multiple can be combined in a single `-o` string.

| Flag | Description |
|------|-------------|
| `merge` | Merge small communities for cache locality |
| `hubx` | Extract high-degree hubs before ordering |
| `gord` | Gorder-inspired intra-community optimization |
| `hsort` | Post-process: pack hubs by descending degree |
| `rcm` | RCM on super-graph instead of dendrogram DFS |
| `norefine` | Disable Leiden refinement step |
| `graphbrew` | Activate LAYER ordering (per-community dispatch) |
| `recursive` | Force recursive sub-community dispatch |
| `flat` | Force flat dispatch (no recursion) |

### Layer 4: GraphBrew Dispatch
When `graphbrew` feature is active: `finalAlgoId` (0-11), `depth` (-1=auto), `subAlgoId`.

### Layer 5: Numeric Parameters
Resolution (float 0.1-3.0), max_iterations, max_passes.

### Example
```bash
-o 12:leiden:hrab:gvecsr:merge:hubx:0.75
# preset=leiden, ordering=hrab, aggregation=gvecsr,
# features=[merge,hubx], resolution=0.75
```

See `GRAPHBREW_LAYERS` in `scripts/lib/utils.py` for the authoritative definition.

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
for algo in 0 4 7 12 14 15; do
    echo "Algorithm $algo:"
    ./bench/bin/pr -f graph.el -s -o $algo -n 5 2>&1 | grep "Average"
done
```

Run the full pipeline on your graphs to measure actual speedups.

See [[Troubleshooting]] for common issues (small communities, worse than HUBCLUSTER, memory).

---

## Next Steps

- [[AdaptiveOrder-ML]] - Automatic algorithm selection per community
- [[Reordering-Algorithms]] - All available algorithms
- [[Running-Benchmarks]] - Benchmark commands

---

[← Back to Home](Home) | [Reordering Algorithms →](Reordering-Algorithms)
