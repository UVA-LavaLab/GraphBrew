# Graph Reordering Algorithms

GraphBrew implements **18 different vertex reordering algorithms** (IDs 0-11, 13, 15-18), each with unique characteristics suited for different graph topologies. This page explains each algorithm in detail.

Note: Algorithm ID 14 (MAP) is reserved for external label mapping files, not a standalone reordering algorithm.

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
| **Community** | RABBITORDER, GORDER, CORDER | Modular graphs |
| **Leiden-Based** | LeidenOrder (15), LeidenDendrogram (16), LeidenCSR (17) | Strong community structure |
| **Hybrid** | GraphBrewOrder (12), AdaptiveOrder (14) | Adaptive selection |

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

## Community-Based Algorithms (8-12)

These algorithms detect **communities** (densely connected subgraphs) and reorder to keep community members together.

### 8. RABBITORDER
**Rabbit Order (community + incremental aggregation)**

```bash
RABBIT_ENABLE=1 make pr
./bench/bin/pr -f graph.el -s -o 8 -n 3
```

- **Description**: Hierarchical community detection with incremental aggregation
- **Complexity**: O(n log n) average
- **Requires**: Build with `RABBIT_ENABLE=1`
- **Best for**: Large graphs with hierarchical community structure

**Key insight**: Uses a "rabbit" metaphor where vertices "hop" to form communities.

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

## Advanced Hybrid Algorithms (13-15)

### 13. GraphBrewOrder
**Per-community reordering**

```bash
# Format: -o 12:<frequency>:<intra_algo>:<resolution>
./bench/bin/pr -f graph.el -s -o 12:10:17 -n 3
```

- **Description**: Runs Leiden, then applies a different algorithm within each community
- **Parameters**:
  - `frequency`: Hub frequency threshold (default: 10)
  - `intra_algo`: Algorithm to use within communities (e.g., 17 = LeidenDendrogram)
  - `resolution`: Leiden resolution parameter (default: 0.75)
- **Best for**: Fine-grained control over per-community ordering

### 14. MAP
**Load mapping from file**

```bash
./bench/bin/pr -f graph.el -s -o 13:mapping.lo -n 3
```

- **Description**: Loads a pre-computed vertex ordering from file
- **File formats**: `.lo` (list order) or `.so` (source order)
- **Best for**: Using externally computed orderings

### 15. AdaptiveOrder ⭐ (ML-powered)
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
| `resolution` | 0.75 | Leiden resolution parameter |
| `min_recurse_size` | 50000 | Minimum community size for recursion |
| `mode` | 0 | 0 = per-community, 1 = full-graph adaptive |

**Operating Modes:**
- **Mode 0 (default)**: Run Leiden → select best algorithm per community
- **Mode 1 (full-graph)**: Skip Leiden → pick single best algorithm for entire graph
- **Multi-level (depth>0)**: Recursively apply AdaptiveOrder to large sub-communities

**See**: [[AdaptiveOrder-ML]] for details on the ML model.

---

## Leiden Variants (16-18)

GraphBrew consolidates Leiden algorithms into three main IDs with parameter-based variant selection for cleaner script sweeping.

### 16. LeidenOrder ⭐
**Leiden community detection via igraph library**

```bash
# Format: 15:resolution
./bench/bin/pr -f graph.el -s -o 14 -n 3                    # Default (res=1.0)
./bench/bin/pr -f graph.el -s -o 15:0.75 -n 3               # Lower resolution
./bench/bin/pr -f graph.el -s -o 15:1.5 -n 3                # Higher resolution
```

- **Description**: State-of-the-art community detection algorithm via igraph
- **Complexity**: O(n log n) average
- **Best for**: Graphs with strong community structure

**Key features:**
- Improves on Louvain algorithm
- Guarantees well-connected communities
- Produces high-quality modularity scores

### 17. LeidenDendrogram
**Leiden community detection with dendrogram traversal**

```bash
# Format: 16:resolution:variant
./bench/bin/pr -f graph.el -s -o 16 -n 3                    # Default (hybrid)
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

### 18. LeidenCSR ⭐ (Fastest)
**Fast CSR-native Leiden (no graph conversion)**

```bash
# Format: 17:resolution:passes:variant
./bench/bin/pr -f graph.el -s -o 17 -n 3                    # Default (hubsort)
./bench/bin/pr -f graph.el -s -o 17:1.0:1:dfs -n 3          # DFS ordering
./bench/bin/pr -f graph.el -s -o 17:1.0:1:bfs -n 3          # BFS ordering
./bench/bin/pr -f graph.el -s -o 17:1.0:1:hubsort -n 3      # Hub-sorted (recommended)
./bench/bin/pr -f graph.el -s -o 17:1.0:2:fast -n 3         # Union-Find + Label Prop
./bench/bin/pr -f graph.el -s -o 17:1.0:1:modularity -n 3   # True Leiden (quality)
```

**Variants:**
| Variant | Description | Speed | Quality |
|---------|-------------|-------|---------|
| `dfs` | Hierarchical DFS | Fast | Good |
| `bfs` | Level-first BFS | Fast | Good |
| `hubsort` | Community + degree sort | **Fastest** | Good |
| `fast` | Union-Find + Label Propagation | Very Fast | Moderate |
| `modularity` | True Leiden with modularity | Slower | **Best** |

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
