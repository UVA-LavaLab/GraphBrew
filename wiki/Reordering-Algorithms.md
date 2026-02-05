# Graph Reordering Algorithms

GraphBrew implements **18 different vertex reordering algorithms** (IDs 0-17), each with unique characteristics suited for different graph topologies. This page explains each algorithm in detail.

Note: Algorithm ID 13 (MAP) is reserved for external label mapping files, not a standalone reordering algorithm.

## Why Reorder Graphs?

Graph algorithms spend significant time accessing memory. When vertices are ordered randomly, memory access patterns are unpredictable, causing **cache misses**. Reordering places frequently co-accessed vertices together in memory, dramatically improving cache utilization.

```
Before Reordering:           After Reordering:
Vertex 1 → 5, 99, 2000       Vertex 1 → 2, 3, 4
Vertex 2 → 8, 1500, 3        Vertex 2 → 1, 3, 5
(scattered neighbors)         (nearby neighbors)
```

## Algorithm Categories

| Category | Algorithms | Best For |
|----------|------------|----------|
| **Basic** | ORIGINAL, RANDOM, SORT | Baseline comparisons |
| **Hub-Based** | HUBSORT, HUBCLUSTER | Power-law graphs |
| **DBG-Based** | DBG, HUBSORTDBG, HUBCLUSTERDBG | Cache locality |
| **Community** | RABBITORDER | Hierarchical communities |
| **Classic** | GORDER, CORDER, RCM | Bandwidth reduction |
| **Leiden-Based** | LeidenOrder (15), LeidenDendrogram (16), LeidenCSR (17) | Strong community structure |
| **Hybrid** | GraphBrewOrder (12), MAP (13), AdaptiveOrder (14) | External/Adaptive selection |

---

## Basic Algorithms (0-2)

### 0. ORIGINAL
**Keep original vertex ordering**

```bash
./bench/bin/pr -f graph.el -s -o 0 -n 3
```

- **Description**: Uses vertices in their original order from the input file
- **Complexity**: O(1) - no reordering
- **Best for**: Baseline comparison, already well-ordered graphs
- **When to use**: Always run this first to establish baseline performance

### 1. RANDOM
**Random vertex permutation**

```bash
./bench/bin/pr -f graph.el -s -o 1 -n 3
```

- **Description**: Randomly shuffles all vertices
- **Complexity**: O(n) where n = number of vertices
- **Best for**: Testing, worst-case scenarios
- **When to use**: Debugging, establishing worst-case baseline

### 2. SORT
**Sort vertices by ID**

```bash
./bench/bin/pr -f graph.el -s -o 2 -n 3
```

- **Description**: Sorts vertices in ascending order by original ID
- **Complexity**: O(n log n)
- **Best for**: Graphs where IDs have locality meaning
- **When to use**: When input has meaningful vertex numbering

---

## Hub-Based Algorithms (3-4)

These algorithms prioritize **high-degree vertices (hubs)** which are accessed frequently.

### 3. HUBSORT
**Sort by degree (hubs first)**

```bash
./bench/bin/pr -f graph.el -s -o 3 -n 3
```

- **Description**: Places high-degree vertices (hubs) at the beginning
- **Complexity**: O(n log n)
- **Rationale**: Hubs are accessed most frequently; placing them together improves cache reuse
- **Best for**: Power-law graphs (social networks, web graphs)

**How it works:**
```
Original:  v1(deg=5), v2(deg=100), v3(deg=2), v4(deg=50)
After:     v2(deg=100), v4(deg=50), v1(deg=5), v3(deg=2)
```

### 4. HUBCLUSTER
**Cluster hubs with their neighbors**

```bash
./bench/bin/pr -f graph.el -s -o 4 -n 3
```

- **Description**: Places each hub followed by its neighbors
- **Complexity**: O(n + m) where m = number of edges
- **Rationale**: When accessing a hub, its neighbors are likely accessed next
- **Best for**: Graphs with hub-and-spoke patterns

**How it works:**
```
Hub v2 has neighbors: v1, v5, v8
Ordering: v2, v1, v5, v8, [next hub], ...
```

---

## DBG-Based Algorithms (5-7)

**Degree-Based Grouping (DBG)** creates "frequency zones" based on access patterns.

### 5. DBG
**Degree-Based Grouping**

```bash
./bench/bin/pr -f graph.el -s -o 5 -n 3
```

- **Description**: Groups vertices by degree into logarithmic buckets
- **Complexity**: O(n)
- **Rationale**: Vertices with similar degrees have similar access frequencies
- **Best for**: General-purpose, works well on most graphs

**Bucket structure:**
```
Bucket 0: degree 1
Bucket 1: degree 2-3
Bucket 2: degree 4-7
Bucket 3: degree 8-15
...
```

### 6. HUBSORTDBG
**HUBSORT within DBG buckets**

```bash
./bench/bin/pr -f graph.el -s -o 6 -n 3
```

- **Description**: First groups by DBG, then sorts each bucket by degree
- **Complexity**: O(n log n)
- **Best for**: Combines benefits of both approaches

### 7. HUBCLUSTERDBG ⭐ (Recommended for power-law)
**HUBCLUSTER within DBG buckets**

```bash
./bench/bin/pr -f graph.el -s -o 7 -n 3
```

- **Description**: First groups by DBG, then clusters hubs with neighbors in each bucket
- **Complexity**: O(n + m)
- **Best for**: Power-law graphs with clear hub structure

### Edge Case Handling (Hub-Based Algorithms)

All hub-based algorithms (HUBSORT, HUBCLUSTER, DBG, HUBSORTDBG, HUBCLUSTERDBG) include guards for empty subgraphs:

```cpp
// Guard against empty graphs (prevents division by zero)
if (num_nodes == 0) {
    return;  // Nothing to reorder
}
const int64_t avgDegree = num_edges / num_nodes;
```

This is important when these algorithms are used as the **final algorithm** in GraphBrewOrder, where community subgraphs may have no internal edges on graphs with extreme structure (e.g., Kronecker graphs).

---

## Community & Classic Algorithms (8-11)

These algorithms use different approaches: RabbitOrder detects communities, while GORDER, CORDER, and RCM focus on bandwidth reduction and cache optimization.

### 8. RABBITORDER
**Rabbit Order (community + incremental aggregation using Louvain)**

```bash
# Format: -o 8[:variant] where variant = csr (default) or boost
./bench/bin/pr -f graph.el -s -o 8 -n 3         # Default: CSR variant (native, fast)
./bench/bin/pr -f graph.el -s -o 8:csr -n 3     # Explicit CSR variant
./bench/bin/pr -f graph.el -s -o 8:boost -n 3   # Original Boost-based variant
```

- **Description**: Hierarchical community detection with incremental aggregation (Louvain-based)
- **Complexity**: O(n log n) average
- **Variants**:
  - `csr` (default): Native CSR implementation - faster, no external dependencies
  - `boost`: Original Boost-based implementation - requires Boost library
- **Note**: RabbitOrder is enabled by default (`RABBIT_ENABLE=1` in Makefile)
- **Best for**: Large graphs with hierarchical community structure
- **Limitation**: Uses Louvain (no refinement), can over-merge communities

**Isolated Vertex Handling**: Both variants group isolated (degree-0) vertices at the end of the permutation, matching Boost's original behavior and improving cache locality for non-isolated vertices.

**Key insight**: Uses a "rabbit" metaphor where vertices "hop" to form communities.

**Comparison with GVE-Leiden (Algorithm 17)**:
| Metric | RabbitOrder | GVE-Leiden |
|--------|-------------|------------|
| Algorithm | Louvain (no refinement) | Leiden (with refinement) |
| Community Quality | Good | Better |
| Speed | Faster | Slightly slower |
| Over-merging | Can occur | Prevented by refinement |

### 9. GORDER
**Graph Ordering (dynamic programming + BFS)**

```bash
./bench/bin/pr -f graph.el -s -o 9 -n 3
```

- **Description**: Uses dynamic programming with sliding window optimization
- **Complexity**: O(n × w) where w = window size
- **Best for**: Graphs where local structure matters

**Window optimization:**
```
Window size determines how far ahead to look when placing vertices
Larger window = better quality, slower computation
```

### 10. CORDER
**Cache-aware Ordering**

```bash
./bench/bin/pr -f graph.el -s -o 10 -n 3
```

- **Description**: Explicitly optimizes for CPU cache hierarchy
- **Complexity**: O(n + m)
- **Best for**: Cache-sensitive applications

### 11. RCM
**Reverse Cuthill-McKee**

```bash
./bench/bin/pr -f graph.el -s -o 11 -n 3
```

- **Description**: Classic bandwidth-reduction algorithm (BFS-based)
- **Complexity**: O(n + m)
- **Best for**: Sparse matrices, scientific computing graphs
- **Note**: Originally designed for sparse matrix solvers

**How it works:**
1. Start from a peripheral vertex (far from center)
2. BFS traversal, ordering by increasing degree
3. Reverse the final ordering

---

## Advanced Hybrid Algorithms (12-14)

### 12. GraphBrewOrder
**Per-community reordering with variant support**

```bash
# Format: -o 12[:variant[:frequency[:intra_algo[:resolution[:maxIterations[:maxPasses]]]]]]
./bench/bin/pr -f graph.el -s -o 12 -n 3                   # Use defaults (leiden variant)
./bench/bin/pr -f graph.el -s -o 12:leiden -n 3            # Explicit leiden variant
./bench/bin/pr -f graph.el -s -o 12:gve -n 3               # GVE-Leiden variant (faster)
./bench/bin/pr -f graph.el -s -o 12:gveopt -n 3            # Cache-optimized GVE
./bench/bin/pr -f graph.el -s -o 12:gve:10:8 -n 3          # gve variant, freq=10, intra=RabbitOrder
```

- **Description**: Runs community detection, then applies per-community reordering
- **Variants**:
  - `leiden`: Original Leiden library (igraph-based) - **default**
  - `gve`: GVE-Leiden CSR-native implementation
  - `gveopt`: Cache-optimized GVE with prefetching
  - `gvefast`: Single-pass GVE (faster, less refinement)
  - `gveoptfast`: Cache-optimized single-pass GVE
  - `rabbit`: RabbitOrder-based community detection
  - `hubcluster`: Hub-clustering based approach
- **Parameters**:
  - `frequency`: Hub frequency threshold (default: 10) - controls how edges are categorized
  - `intra_algo`: Algorithm ID to use within communities (default: 8 = RabbitOrder)
  - `resolution`: Leiden resolution parameter (default: dynamic for best PR performance)
  - `maxIterations`: Maximum Leiden iterations (default: 30)
  - `maxPasses`: Maximum Leiden passes (default: 30)
- **Dynamic thresholds**: Community size thresholds are computed dynamically based on `avg_community_size/4` and `sqrt(N)`
- **Best for**: Fine-grained control over per-community ordering

### 13. MAP
**Load mapping from file**

```bash
./bench/bin/pr -f graph.el -s -o 13:mapping.lo -n 3
```

- **Description**: Loads a pre-computed vertex ordering from file
- **File formats**: `.lo` (list order) or `.so` (source order)
- **Best for**: Using externally computed orderings

### 14. AdaptiveOrder ⭐ (ML-powered)
**Perceptron-based algorithm selection**

```bash
# Format: -o 14[:max_depth[:resolution[:min_recurse_size[:mode]]]]

# Default: per-community selection
./bench/bin/pr -f graph.el -s -o 14 -n 3

# Multi-level: recurse into large communities (depth=2)
./bench/bin/pr -f graph.el -s -o 14:2 -n 3

# Full-graph mode: pick single best algorithm for entire graph
./bench/bin/pr -f graph.el -s -o 14:0:0.75:50000:1 -n 3
```

- **Description**: Uses ML to select the best algorithm for each community
- **Complexity**: O(n log n) + perceptron inference
- **Best for**: Unknown graphs, automated pipelines

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 0 | Max recursion depth (0 = per-community, 1+ = multi-level) |
| `resolution` | auto | Leiden resolution (auto: continuous formula with CV guardrail) |
| `min_recurse_size` | 50000 | Minimum community size for recursion |
| `mode` | 0 | 0 = per-community, 1 = full-graph adaptive |

**Auto-Resolution Formula:**
```
γ = clip(0.5 + 0.25 × log₁₀(avg_degree + 1), 0.5, 1.2)
If CV(degree) > 2: γ = max(γ, 1.0)  // CV guardrail for hubby graphs
```
*Heuristic for stable partitions; users should sweep γ for best community quality.*

**Operating Modes:**
- **Mode 0 (default)**: Run Leiden → select best algorithm per community
- **Mode 1 (full-graph)**: Skip Leiden → pick single best algorithm for entire graph
- **Multi-level (depth>0)**: Recursively apply AdaptiveOrder to large sub-communities

**See**: [[AdaptiveOrder-ML]] for details on the ML model.

---

## Leiden Variants (15-17)

GraphBrew consolidates Leiden algorithms into three main IDs with parameter-based variant selection for cleaner script sweeping.

### 15. LeidenOrder ⭐
**Leiden community detection via igraph library**

```bash
# Format: -o 15:resolution
./bench/bin/pr -f graph.el -s -o 15 -n 3                    # Default (auto-resolution)
./bench/bin/pr -f graph.el -s -o 15:0.75 -n 3               # Lower resolution
./bench/bin/pr -f graph.el -s -o 15:1.5 -n 3                # Higher resolution
```

- **Description**: State-of-the-art community detection algorithm via igraph
- **Complexity**: O(n log n) average
- **Best for**: Graphs with strong community structure
- **Default resolution**: Auto-detected via continuous formula (0.5-1.2) with CV guardrail for power-law graphs

**Key features:**
- Improves on Louvain algorithm
- Guarantees well-connected communities
- Produces high-quality modularity scores

### 16. LeidenDendrogram
**Leiden community detection with dendrogram traversal**

```bash
# Format: -o 16:variant:resolution
./bench/bin/pr -f graph.el -s -o 16 -n 3                    # Default (auto-resolution, hybrid)
./bench/bin/pr -f graph.el -s -o 16:dfs:1.0 -n 3            # DFS traversal
./bench/bin/pr -f graph.el -s -o 16:dfshub:1.0 -n 3         # DFS hub-first
./bench/bin/pr -f graph.el -s -o 16:dfssize:1.0 -n 3        # DFS size-first
./bench/bin/pr -f graph.el -s -o 16:bfs:1.0 -n 3            # BFS traversal
./bench/bin/pr -f graph.el -s -o 16:hybrid:1.0 -n 3         # Hybrid (recommended)
```

**Variants:**
| Variant | Description | Best For |
|---------|-------------|----------|
| `dfs` | Standard DFS traversal | General hierarchical |
| `dfshub` | DFS with hub-first ordering | Power-law graphs |
| `dfssize` | DFS with size-first ordering | Uneven community sizes |
| `bfs` | BFS level-order traversal | Wide hierarchies |
| `hybrid` | Sort by (community, degree) | **Default - best overall** |

### 17. LeidenCSR ⭐ (Fastest, Best Quality)
**GVE-Leiden: Fast CSR-native Leiden with proper refinement**

Implements the full Leiden algorithm from: *"Fast Leiden Algorithm for Community Detection in Shared Memory Setting"* (ACM DOI 10.1145/3673038.3673146)

```bash
# Format: -o 17[:variant:resolution:iterations:passes]
./bench/bin/pr -f graph.el -s -o 17 -n 3                    # Default: GVE-Leiden (best quality)
./bench/bin/pr -f graph.el -s -o 17:gve:1.0:20:5 -n 3       # GVE-Leiden explicit params
./bench/bin/pr -f graph.el -s -o 17:gveopt:auto -n 3        # Cache-optimized GVE, auto resolution
./bench/bin/pr -f graph.el -s -o 17:gveopt2:2.0:20:10 -n 3  # CSR-based aggregation (fastest)
./bench/bin/pr -f graph.el -s -o 17:gveadaptive:dynamic -n 3 # Dynamic resolution adjustment
./bench/bin/pr -f graph.el -s -o 17:gvedendo:1.0:20:5 -n 3  # GVE with incremental dendrogram
./bench/bin/pr -f graph.el -s -o 17:gveoptsort:auto -n 3    # Multi-level sort ordering
./bench/bin/pr -f graph.el -s -o 17:gveturbo:1.0:5:10 -n 3  # Speed-optimized (skip refinement)
./bench/bin/pr -f graph.el -s -o 17:gverabbit:1.0:5 -n 3    # GVE-Rabbit hybrid (fast)
./bench/bin/pr -f graph.el -s -o 17:hubsort:1.0:10:3 -n 3   # Hub-sorted variant
./bench/bin/pr -f graph.el -s -o 17:fast:1.0:10:2 -n 3      # Union-Find + Label Prop
./bench/bin/pr -f graph.el -s -o 17:dfs:1.0:10:1 -n 3       # DFS ordering
./bench/bin/pr -f graph.el -s -o 17:bfs:1.0:10:1 -n 3       # BFS ordering
```

**Resolution Parameter Modes:**

| Mode | Syntax | Description |
|------|--------|-------------|
| **Fixed** | `1.5` | Use specified resolution value |
| **Auto** | `auto` or `0` | Compute from graph density and CV |
| **Dynamic** | `dynamic` | Auto initial, adjust per-pass (gveadaptive only) |
| **Dynamic+Initial** | `dynamic_2.0` | Start at 2.0, adjust per-pass |

```bash
# Resolution modes examples
./bench/bin/pr -f graph.el -s -o 17:gveopt2:1.5 -n 3       # Fixed resolution
./bench/bin/pr -f graph.el -s -o 17:gveopt2:auto -n 3      # Auto-computed resolution
./bench/bin/pr -f graph.el -s -o 17:gveopt2:0 -n 3         # Same as auto
./bench/bin/pr -f graph.el -s -o 17:gveadaptive:dynamic -n 3      # Dynamic adjustment
./bench/bin/pr -f graph.el -s -o 17:gveadaptive:dynamic_2.0 -n 3  # Dynamic, start at 2.0
```

**Variants:**
| Variant | Description | Speed | Quality | Best For |
|---------|-------------|-------|---------|----------|
| `gve` | **GVE-Leiden with refinement (DEFAULT)** | Fast | **Best** | General use |
| `gveopt` | Cache-optimized GVE with prefetching | **Faster** | **Best** | Large graphs |
| `gveopt2` | **CSR-based aggregation (no sort)** | **Fastest** | **Best** | Best overall ⭐ |
| `gveadaptive` | **Dynamic resolution adjustment** | Fast | **Best** | Unknown graphs ⭐ |
| `gveoptsort` | Multi-level sort ordering (LeidenOrder-style) | Fast | **Best** | Hierarchical |
| `gveturbo` | Speed-optimized (optional refinement skip) | **Fastest** | Good | Speed priority |
| `gvedendo` | GVE with incremental dendrogram | Fast | **Best** | Dendrogram needs |
| `gveoptdendo` | GVEopt with incremental dendrogram | **Faster** | **Best** | Dendrogram needs |
| `gvefast` | CSR buffer reuse (leiden.hxx style) | **Fastest** | **Best** | Large graphs |
| `gverabbit` | GVE-Rabbit hybrid (limited iterations) | **Fastest** | Good | Very large graphs |
| `dfs` | Hierarchical DFS | Fast | Good | Tree structures |
| `bfs` | Level-first BFS | Fast | Good | Wide hierarchies |
| `hubsort` | Community + degree sort | Fast | Good | Power-law graphs |
| `fast` | Union-Find + Label Propagation | Very Fast | Moderate | Speed priority |
| `modularity` | Modularity optimization | Fast | Good | Quality focus |

**New Optimized Variants (GVEOpt2 & GVEAdaptive)**:

**GVEOpt2** - CSR-based aggregation replaces O(E log E) sort with O(E) community-first scanning:
```bash
# Auto-resolution (recommended for unknown graphs)
./bench/bin/pr -f graph.mtx -s -o 17:gveopt2:auto -n 3

# Fixed resolution (1.5-2.0 often best for social networks)
./bench/bin/pr -f graph.mtx -s -o 17:gveopt2:2.0 -n 3
```

**GVEAdaptive** - Dynamically adjusts resolution at each pass based on runtime metrics:
```bash
# Dynamic mode (auto initial, adjusts each pass)
./bench/bin/pr -f graph.mtx -s -o 17:gveadaptive:dynamic -n 3

# Dynamic mode with initial resolution 2.0
./bench/bin/pr -f graph.mtx -s -o 17:gveadaptive:dynamic_2.0 -n 3
```

The adaptive algorithm monitors 4 signals each pass:
1. **Community reduction rate** - Too fast? Raise resolution. Too slow? Lower it.
2. **Size imbalance** - Giant communities? Raise resolution to break them.
3. **Convergence speed** - 1 iteration? Communities too stable, raise resolution.
4. **Super-graph density** - Denser graphs need higher resolution.

**Resolution and Cache Locality Trade-off**:
| Resolution | Modularity | Communities | PR Speed | Best For |
|------------|-----------|-------------|----------|----------|
| Low (0.3-0.5) | High (0.66-0.74) | Few large | Slow | True community detection |
| High (1.0-2.0) | Lower (0.55-0.59) | Many small | **Fast** | Cache locality / reordering |

*Key insight*: For graph reordering, we optimize for **cache locality**, not sociological correctness. Higher resolution creates more balanced communities that fit better in cache.

**Benchmark Results (wiki-Talk: 2.4M nodes, 5M edges)**:
| Variant | Reorder Time | PR Execution | vs LeidenOrder |
|---------|--------------|--------------|----------------|
| LeidenOrder (15) | 1.62s | 0.042s | baseline |
| GVEOpt2 res=2.0 | 1.29s | **0.033s** | **21% faster PR** |
| GVEAdaptive res=2.0 | **1.14s** | 0.035s | **30% faster reorder** |
| GVEOpt | 1.83s | 0.077s | slower |
| GVETurbo | 1.24s | 0.093s | fast reorder, slow PR |

**GVE-Dendo Variants**:
The `gvedendo` and `gveoptdendo` variants implement incremental dendrogram building inspired by RabbitOrder's approach:
- **Standard GVE**: Stores `community_per_pass` history and rebuilds the dendrogram tree in post-processing
- **GVE-Dendo**: Builds parent-child relationships incrementally during the refinement phase using atomic operations
- Benefits: Avoids post-processing tree reconstruction, preserves same modularity quality

**GVE-Leiden Algorithm (3-phase)**:
1. **Phase 1: Local-moving** - Greedily move vertices to maximize modularity
2. **Phase 2: Refinement** - Only allow isolated vertices to move, ensuring well-connected communities
3. **Phase 3: Aggregation** - Build super-graph and repeat hierarchically

**Isolated Vertex Handling**: Degree-0 vertices are automatically identified and grouped at the end of the permutation. This improves cache locality for active vertices during graph traversals. The algorithm reports the number of isolated vertices found.

**Why GVE-Leiden beats RabbitOrder (Louvain)**:
| Graph Type | RabbitOrder Q | GVE-Leiden Q | Improvement |
|------------|---------------|--------------|-------------|
| Web graphs | 0.977 | 0.983 | +0.6% |
| Road networks | 0.988 | 0.992 | +0.4% |
| Social networks | 0.650 | 0.788 | +21% |
| Synthetic (Kronecker) | 0.063 | 0.190 | +3x |

**Comprehensive Variant Benchmark (as-Skitter: 1.7M nodes, 11M edges)**:
| Variant | Reorder(s) | PR Time(s) | Modularity | Communities |
|---------|------------|------------|------------|-------------|
| gve | 3.70 | 0.082 | 0.881 | 12 |
| gveopt | 1.95 | 0.090 | 0.892 | 675 |
| **gveopt2** | **1.44** | **0.070** | 0.858 | 1,712 |
| gveadaptive | 1.55 | 0.138 | 0.857 | 2,066 |
| gveoptsort | 1.72 | 0.095 | 0.893 | 1,378 |
| **gveturbo** | **0.77** | 0.076 | 0.873 | 1,480 |

*Key insight*: For **speed priority**, use `gveturbo` or `gveopt2`. For **quality priority**, use `gve` or `gveopt`.

**Sweeping Variants Example:**
```bash
# Sweep all LeidenCSR variants (format: 17:variant:resolution:iterations:passes)
for variant in gve gveopt gveopt2 gveadaptive gveoptsort gveturbo gvefast gverabbit; do
    ./bench/bin/pr -f graph.mtx -s -o 17:$variant:1.0:10:10 -n 5
done

# Resolution sweep for optimal cache locality
for res in 0.5 1.0 1.5 2.0; do
    ./bench/bin/pr -f graph.mtx -s -o 17:gveopt2:$res -n 5
done
```

### VIBE: Unified Reordering Framework

**VIBE (Vertex Indexing for Better Efficiency)** provides a unified interface for graph reordering with two main algorithms and configurable ordering strategies. All VIBE variants use the unified `reorder::ReorderConfig` defaults:

- **Resolution**: Auto-computed from graph properties (density, degree distribution)
- **Max Iterations**: 10 per pass
- **Max Passes**: 10 total
- **Dynamic Resolution**: Optional per-pass adjustment based on runtime metrics

```bash
# Format: -o 17:vibe[:algorithm][:ordering][:aggregation][:resolution_mode]

# Leiden-based VIBE (multi-pass community detection)
./bench/bin/pr -f graph.mtx -s -o 17:vibe -n 3            # Hierarchical ordering (default)
./bench/bin/pr -f graph.mtx -s -o 17:vibe:dfs -n 3        # DFS dendrogram traversal
./bench/bin/pr -f graph.mtx -s -o 17:vibe:bfs -n 3        # BFS dendrogram traversal
./bench/bin/pr -f graph.mtx -s -o 17:vibe:dbg -n 3        # DBG within each community
./bench/bin/pr -f graph.mtx -s -o 17:vibe:corder -n 3     # Hot/cold within communities
./bench/bin/pr -f graph.mtx -s -o 17:vibe:dbg-global -n 3 # DBG across all vertices
./bench/bin/pr -f graph.mtx -s -o 17:vibe:streaming -n 3  # Lazy aggregation (faster)
./bench/bin/pr -f graph.mtx -s -o 17:vibe:lazyupdate -n 3 # Batched ctot updates (reduces atomics)
./bench/bin/pr -f graph.mtx -s -o 17:vibe:conn -n 3       # Connectivity BFS within communities (default ordering)
./bench/bin/pr -f graph.mtx -s -o 17:vibe:hrab -n 3       # Hybrid Leiden+RabbitOrder (best for web/geometric)

# Resolution modes
./bench/bin/pr -f graph.mtx -s -o 17:vibe:auto -n 3       # Auto (graph-adaptive, computed once)
./bench/bin/pr -f graph.mtx -s -o 17:vibe:dynamic -n 3    # Dynamic (adjusted per-pass)
./bench/bin/pr -f graph.mtx -s -o 17:vibe:0.75 -n 3       # Fixed resolution 0.75

# RabbitOrder-based VIBE (single-pass parallel aggregation)
./bench/bin/pr -f graph.mtx -s -o 17:vibe:rabbit -n 3           # RabbitOrder (DFS default)
./bench/bin/pr -f graph.mtx -s -o 17:vibe:rabbit:dfs -n 3       # + DFS post-ordering
./bench/bin/pr -f graph.mtx -s -o 17:vibe:rabbit:bfs -n 3       # + BFS post-ordering
./bench/bin/pr -f graph.mtx -s -o 17:vibe:rabbit:dbg -n 3       # + DBG post-ordering
./bench/bin/pr -f graph.mtx -s -o 17:vibe:rabbit:corder -n 3    # + COrder post-ordering
```

**VIBE Resolution Modes:**

| Mode | Option | Description |
|------|--------|-------------|
| Auto | `vibe:auto` or `vibe` | Computed once from graph properties (default) |
| Dynamic | `vibe:dynamic` | Adjusted per-pass based on runtime metrics |
| Fixed | `vibe:0.75` | User-specified fixed value |

**VIBE Algorithm Comparison:**

| Aspect | `vibe` (Leiden) | `vibe:rabbit` (RabbitOrder) |
|--------|-----------------|----------------------------|
| **Passes** | Multi-pass (2-5) | Single-pass |
| **Vertex Order** | Random/parallel | Sorted by degree (ascending) |
| **Aggregation** | Explicit super-graph or lazy | Implicit union-find + edge cache |
| **Parallelism** | Per-pass parallel | Lock-free 64-bit CAS |
| **Dendrogram** | Built after detection | Built during merges |
| **Communities** | Many fine (~50K) | Fewer coarse (~2K) |
| **Ordering** | Configurable | DFS (configurable post) |
| **Best For** | Quality communities | Fast reordering |

**VIBE Ordering Strategies (for Leiden-based):**

| Strategy | Option | Description |
|----------|--------|-------------|
| HIERARCHICAL | `vibe` | Sort by community, then by degree |
| DENDROGRAM_DFS | `vibe:dfs` | DFS traversal of dendrogram |
| DENDROGRAM_BFS | `vibe:bfs` | BFS traversal of dendrogram |
| DBG | `vibe:dbg` | DBG algorithm within each community |
| CORDER | `vibe:corder` | Hot/cold separation within communities |
| DBG_GLOBAL | `vibe:dbg-global` | DBG across all vertices (post-clustering) |
| CORDER_GLOBAL | `vibe:corder-global` | Hot/cold across all vertices |
| CONNECTIVITY_BFS | `vibe:conn` | BFS within communities using original graph edges (Boost-style, default) |
| HYBRID_LEIDEN_RABBIT | `vibe:hrab` | Leiden communities + RabbitOrder super-graph ordering (best locality) |

**New Ordering Strategies (v2):**

**`vibe:conn` — Connectivity BFS** (aliases: `connbfs`, `connectivity`)
BFS within each Leiden community using the original graph's adjacency structure. Unlike hierarchical ordering (which sorts by community ID + degree), connectivity BFS follows actual edges to place connected vertices consecutively. This is the approach used by RabbitOrder's Boost implementation. It serves as the default ordering strategy.

**`vibe:hrab` — Hybrid Leiden+RabbitOrder** (aliases: `hybrid-rabbit`, `leidenrabbit`)
A two-phase approach combining Leiden's high-quality community detection with RabbitOrder's super-graph ordering:
1. **Phase 1: Leiden community detection** — Detects communities using GVE-Leiden (multi-pass refinement)
2. **Phase 2: Super-graph construction** — Builds a weighted graph where each node is a community, edge weights = inter-community edges
3. **Phase 3: RabbitOrder on super-graph** — Orders communities using RabbitOrder's cache-aware aggregation
4. **Phase 4: BFS within communities** — Orders vertices within each community using connectivity BFS

This hybrid approach captures the best of both worlds: Leiden's superior modularity quality (especially on power-law graphs) and RabbitOrder's cache-aware ordering of the community hierarchy. Benchmarks show **31% better geometric mean cache miss distance** compared to standalone RabbitOrder on web graphs (indochina-2004), and **96.7% near-neighbor hit rate** on geometric graphs (rgg_n_2_24_s0).

**VIBE vs 8:csr (Native Rabbit) Benchmark:**

| Graph | 8:csr Reorder | 8:csr PR | vibe:rabbit Reorder | vibe:rabbit PR |
|-------|---------------|----------|---------------------|----------------|
| web-Google | 0.22s | 0.016s | 0.22s | 0.015s |
| wiki-Talk | 0.47s | 0.043s | **0.43s** | **0.038s** |
| soc-Epinions1 | 0.04s | 0.008s | **0.03s** | **0.007s** |
| roadNet-CA | 0.29s | 0.018s | 0.30s | 0.018s |
| cit-Patents | **1.01s** | 0.123s | 1.07s | **0.115s** |
| web-BerkStan | 0.27s | 0.030s | **0.13s** | **0.026s** |

---

## Algorithm Selection Guide

### By Graph Type

| Graph Type | Recommended | Alternatives |
|------------|-------------|--------------|
| Social Networks | LeidenCSR (17:gveopt2) | LeidenDendrogram (16:hybrid) |
| Web Graphs | VIBE (17:vibe:hrab) | LeidenCSR (17:gveopt2) |
| Road Networks | ORIGINAL (0), RCM (11) | LeidenCSR (17:gveadaptive) |
| Citation Networks | LeidenCSR (17:gve) | LeidenOrder (15) |
| Random Geometric | VIBE (17:vibe:hrab) | LeidenCSR (17:gveopt2) |
| Unknown | LeidenCSR (17:gveadaptive) | AdaptiveOrder (14) |

### By Graph Size

| Size | Nodes | Recommended |
|------|-------|-------------|
| Small | < 100K | Any (try several) |
| Medium | 100K - 1M | LeidenCSR (17:gveopt2) |
| Large | 1M - 100M | LeidenCSR (17:gveopt2), LeidenCSR (17:gveturbo) |
| Very Large | > 100M | LeidenCSR (17:gveturbo), HUBCLUSTERDBG (7) |

### Quick Decision Tree

```
Is your graph modular (has communities)?
├── Yes → Is it very large (>10M vertices)?
│         ├── Yes → LeidenCSR (17:fast) for speed
│         │         LeidenCSR (17:gve) for quality
│         └── No → LeidenCSR (17) or (17:gve) - best quality
└── No/Unknown → Is it a power-law graph?
              ├── Yes → HUBCLUSTERDBG (7)
              └── No → Try AdaptiveOrder (14)
```

---

## Performance Comparison Example

Running PageRank on a social network (1M vertices, 10M edges):

| Algorithm | Time | Speedup |
|-----------|------|---------|
| ORIGINAL (0) | 1.00s | 1.00x |
| RANDOM (1) | 1.45s | 0.69x |
| HUBSORT (3) | 0.85s | 1.18x |
| DBG (5) | 0.80s | 1.25x |
| HUBCLUSTERDBG (7) | 0.72s | 1.39x |
| RabbitOrder (8) | 0.68s | 1.47x |
| LeidenOrder (15) | 0.65s | 1.54x |
| LeidenCSR (17:gve) | 0.55s | 1.82x |

---

## Next Steps

- [[Running-Benchmarks]] - How to run experiments
- [[AdaptiveOrder-ML]] - Deep dive into ML-based selection
- [[Adding-New-Algorithms]] - Implement your own algorithm

---

[← Back to Home](Home)
