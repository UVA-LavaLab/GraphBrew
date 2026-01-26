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

---

## Community & Classic Algorithms (8-11)

These algorithms use different approaches: RabbitOrder detects communities, while GORDER, CORDER, and RCM focus on bandwidth reduction and cache optimization.

### 8. RABBITORDER
**Rabbit Order (community + incremental aggregation using Louvain)**

```bash
./bench/bin/pr -f graph.el -s -o 8 -n 3
```

- **Description**: Hierarchical community detection with incremental aggregation (Louvain-based)
- **Complexity**: O(n log n) average
- **Note**: RabbitOrder is enabled by default (`RABBIT_ENABLE=1` in Makefile)
- **Best for**: Large graphs with hierarchical community structure
- **Limitation**: Uses Louvain (no refinement), can over-merge communities

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
**Per-community reordering**

```bash
# Format: -o 12[:frequency[:intra_algo[:resolution[:maxIterations[:maxPasses]]]]]
./bench/bin/pr -f graph.el -s -o 12 -n 3                # Use defaults (auto-resolution)
./bench/bin/pr -f graph.el -s -o 12:10 -n 3             # frequency=10
./bench/bin/pr -f graph.el -s -o 12:10:8 -n 3           # frequency=10, intra_algo=8 (RabbitOrder)
./bench/bin/pr -f graph.el -s -o 12:10:16:0.75 -n 3     # frequency=10, intra=16, res=0.75
```

- **Description**: Runs Leiden, then applies a different algorithm within each community
- **Parameters**:
  - `frequency`: Hub frequency threshold (default: 10) - controls how edges are categorized
  - `intra_algo`: Algorithm ID to use within communities (default: 8 = RabbitOrder)
  - `resolution`: Leiden resolution parameter (default: auto based on density)
  - `maxIterations`: Maximum Leiden iterations (default: 30)
  - `maxPasses`: Maximum Leiden passes (default: 30)
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
# Format: -o 16:resolution:variant
./bench/bin/pr -f graph.el -s -o 16 -n 3                    # Default (auto-resolution, hybrid)
./bench/bin/pr -f graph.el -s -o 16:1.0:dfs -n 3            # DFS traversal
./bench/bin/pr -f graph.el -s -o 16:1.0:dfshub -n 3         # DFS hub-first
./bench/bin/pr -f graph.el -s -o 16:1.0:dfssize -n 3        # DFS size-first
./bench/bin/pr -f graph.el -s -o 16:1.0:bfs -n 3            # BFS traversal
./bench/bin/pr -f graph.el -s -o 16:1.0:hybrid -n 3         # Hybrid (recommended)
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
# Format: -o 17:variant:resolution:passes
./bench/bin/pr -f graph.el -s -o 17 -n 3                    # Default: gve variant
./bench/bin/pr -f graph.el -s -o 17:gve:1.0:20 -n 3         # GVE-Leiden (recommended)
./bench/bin/pr -f graph.el -s -o 17:dfs:1.0:1 -n 3          # DFS ordering
./bench/bin/pr -f graph.el -s -o 17:hubsort:1.0:1 -n 3      # Hub-sorted
./bench/bin/pr -f graph.el -s -o 17:fast:1.0:2 -n 3         # Union-Find + Label Prop
```

**Variants:**
| Variant | Description | Speed | Quality |
|---------|-------------|-------|---------|
| `gve` | **GVE-Leiden with refinement** | Fast | **Best** |
| `dfs` | Hierarchical DFS | Fast | Good |
| `bfs` | Level-first BFS | Fast | Good |
| `hubsort` | Community + degree sort | Fastest | Good |
| `fast` | Union-Find + Label Propagation | Very Fast | Moderate |

**GVE-Leiden Algorithm (3-phase)**:
1. **Phase 1: Local-moving** - Greedily move vertices to maximize modularity
2. **Phase 2: Refinement** - Only allow isolated vertices to move, ensuring well-connected communities
3. **Phase 3: Aggregation** - Build super-graph and repeat hierarchically

**Why GVE-Leiden beats RabbitOrder (Louvain)**:
| Graph Type | RabbitOrder Q | GVE-Leiden Q | Improvement |
|------------|---------------|--------------|-------------|
| Web graphs | 0.977 | 0.983 | +0.6% |
| Road networks | 0.988 | 0.992 | +0.4% |
| Social networks | 0.650 | 0.788 | +21% |
| Synthetic (Kronecker) | 0.063 | 0.190 | +3x |

**Sweeping Variants Example:**
```bash
# Sweep all LeidenCSR variants
for variant in dfs bfs hubsort fast modularity; do
    ./bench/bin/pr -f graph.mtx -s -o 17:1.0:1:$variant -n 5
done
```

---

## Algorithm Selection Guide

### By Graph Type

| Graph Type | Recommended | Alternatives |
|------------|-------------|--------------|
| Social Networks | LeidenDendrogram (16:hybrid) | LeidenCSR (17:hubsort) |
| Web Graphs | LeidenCSR (17:hubsort) | HUBCLUSTERDBG (7) |
| Road Networks | ORIGINAL (0), RCM (11) | GORDER (9) |
| Citation Networks | LeidenOrder (15) | RABBITORDER (8) |
| Unknown | AdaptiveOrder (14) | LeidenCSR (17) |

### By Graph Size

| Size | Nodes | Recommended |
|------|-------|-------------|
| Small | < 100K | Any (try several) |
| Medium | 100K - 1M | LeidenCSR (17:hubsort) |
| Large | 1M - 100M | LeidenCSR (17:hubsort), AdaptiveOrder (14) |
| Very Large | > 100M | HUBCLUSTERDBG (7), LeidenCSR (17:fast) |

### Quick Decision Tree

```
Is your graph modular (has communities)?
├── Yes → Is it very large (>10M vertices)?
│         ├── Yes → LeidenCSR (17:fast) for speed
│         │         LeidenCSR (17:modularity) for quality
│         └── No → LeidenCSR (17:hubsort)
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
| LeidenOrder (15) | 0.65s | 1.54x |
| LeidenCSR (17:hubsort) | 0.58s | 1.72x |

---

## Next Steps

- [[Running-Benchmarks]] - How to run experiments
- [[AdaptiveOrder-ML]] - Deep dive into ML-based selection
- [[Adding-New-Algorithms]] - Implement your own algorithm

---

[← Back to Home](Home)
