# Graph Reordering Algorithms

GraphBrew implements **20 different vertex reordering algorithms** (IDs 0-13 and 15-20), each with unique characteristics suited for different graph topologies. This page explains each algorithm in detail.

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
| **Leiden-Based** | LeidenOrder, LeidenHybrid, etc. | Strong community structure |
| **Hybrid** | GraphBrewOrder, AdaptiveOrder | Adaptive selection |

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

### 12. LeidenOrder ⭐ (Recommended starting point)
**Leiden community detection**

```bash
./bench/bin/pr -f graph.el -s -o 12 -n 3
```

- **Description**: State-of-the-art community detection algorithm
- **Complexity**: O(n log n) average
- **Best for**: Graphs with strong community structure

**Key features:**
- Improves on Louvain algorithm
- Guarantees well-connected communities
- Produces high-quality modularity scores

---

## Advanced Hybrid Algorithms (13-20)

### 13. GraphBrewOrder
**Per-community reordering**

```bash
# Format: -o 13:<frequency>:<intra_algo>:<resolution>
./bench/bin/pr -f graph.el -s -o 13:10:17 -n 3
```

- **Description**: Runs Leiden, then applies a different algorithm within each community
- **Parameters**:
  - `frequency`: Hub frequency threshold (default: 10)
  - `intra_algo`: Algorithm to use within communities (e.g., 17 = LeidenDFSHub)
  - `resolution`: Leiden resolution parameter (default: 0.75)
- **Best for**: Fine-grained control over per-community ordering

### 14. MAP
**Load mapping from file**

```bash
./bench/bin/pr -f graph.el -s -o 14:mapping.lo -n 3
```

- **Description**: Loads a pre-computed vertex ordering from file
- **File formats**: `.lo` (list order) or `.so` (source order)
- **Best for**: Using externally computed orderings

### 15. AdaptiveOrder ⭐ (ML-powered)
**Perceptron-based algorithm selection**

```bash
./bench/bin/pr -f graph.el -s -o 15 -n 3
```

- **Description**: Uses ML to select the best algorithm for each community
- **Complexity**: O(n log n) + perceptron inference
- **Best for**: Unknown graphs, automated pipelines

**How it works:**
1. Run Leiden to detect communities
2. Compute features for each community (size, density, hub concentration)
3. Use trained perceptron to select best algorithm per community
4. Apply selected algorithm to each community

**See**: [[AdaptiveOrder-ML]] for details on the ML model.

---

## Leiden Dendrogram Variants (16-20)

These algorithms use **Leiden community detection** combined with different **dendrogram traversal strategies**.

### What's a Dendrogram?

Leiden produces a hierarchical tree of communities:
```
        Root
       /    \
    Comm1   Comm2
    /  \      |
  C1a  C1b   C2a
```

Different traversal orders produce different vertex orderings.

### 16. LeidenDFS
**Depth-First Search traversal**

```bash
./bench/bin/pr -f graph.el -s -o 16 -n 3
```

- **Description**: Standard DFS traversal of community hierarchy
- **Order**: Goes deep into one branch before exploring siblings
- **Best for**: General hierarchical structure

### 17. LeidenDFSHub
**DFS prioritizing hub communities**

```bash
./bench/bin/pr -f graph.el -s -o 17 -n 3
```

- **Description**: DFS that visits high-degree communities first
- **Rationale**: Hub communities are accessed more frequently
- **Best for**: Power-law graphs

### 18. LeidenDFSSize
**DFS prioritizing larger communities**

```bash
./bench/bin/pr -f graph.el -s -o 18 -n 3
```

- **Description**: DFS that visits larger communities first
- **Rationale**: Larger communities contain more vertices to process
- **Best for**: Graphs with uneven community sizes

### 19. LeidenBFS
**Breadth-First Search traversal**

```bash
./bench/bin/pr -f graph.el -s -o 19 -n 3
```

- **Description**: Level-order traversal of community hierarchy
- **Order**: Processes all communities at one level before going deeper
- **Best for**: Wide, shallow community hierarchies

### 20. LeidenHybrid ⭐ (Often best)
**Hybrid hub-aware DFS**

```bash
./bench/bin/pr -f graph.el -s -o 20 -n 3
```

- **Description**: Combines hub prioritization with adaptive traversal
- **Best for**: Most graphs - good default choice

**Why it's often best:**
- Balances hub frequency with community structure
- Adapts traversal based on community characteristics
- Robust across different graph types

---

## Algorithm Selection Guide

### By Graph Type

| Graph Type | Recommended | Alternatives |
|------------|-------------|--------------|
| Social Networks | LeidenHybrid (20) | LeidenDFSHub (17), AdaptiveOrder (15) |
| Web Graphs | LeidenHybrid (20) | HUBCLUSTERDBG (7) |
| Road Networks | ORIGINAL (0), RCM (11) | GORDER (9) |
| Citation Networks | LeidenOrder (12) | RABBITORDER (8) |
| Unknown | AdaptiveOrder (15) | LeidenHybrid (20) |

### By Graph Size

| Size | Nodes | Recommended |
|------|-------|-------------|
| Small | < 100K | Any (try several) |
| Medium | 100K - 1M | LeidenHybrid (20) |
| Large | 1M - 100M | LeidenHybrid (20), AdaptiveOrder (15) |
| Very Large | > 100M | HUBCLUSTERDBG (7), LeidenOrder (12) |

### Quick Decision Tree

```
Is your graph modular (has communities)?
├── Yes → Is it very large (>10M vertices)?
│         ├── Yes → LeidenOrder (12)
│         └── No → LeidenHybrid (20)
└── No/Unknown → Is it a power-law graph?
              ├── Yes → HUBCLUSTERDBG (7)
              └── No → Try AdaptiveOrder (15)
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
| LeidenOrder (12) | 0.65s | 1.54x |
| LeidenHybrid (20) | 0.58s | 1.72x |

---

## Next Steps

- [[Running-Benchmarks]] - How to run experiments
- [[AdaptiveOrder-ML]] - Deep dive into ML-based selection
- [[Adding-New-Algorithms]] - Implement your own algorithm

---

[← Back to Home](Home)
