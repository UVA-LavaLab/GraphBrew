# Community Detection with Leiden

Understanding how GraphBrew uses Leiden community detection for graph reordering.

---

## What is Community Detection?

Community detection finds **densely connected groups** of vertices in a graph. Vertices within a community have many connections to each other and fewer connections to vertices outside.

```
Before:                    After Leiden:
  1---2                    Community A: {1,2,3}
  |\ /|                    Community B: {4,5,6}
  | X |
  |/ \|                    
  3   4---5
      |\ /|
      | X |
      |/ \|
      6
```

---

## Why Use Community Detection for Reordering?

### Cache Locality

When vertices from the same community are numbered consecutively:
- Their data is contiguous in memory
- Cache lines are used efficiently
- Fewer cache misses during graph traversal

```
Original IDs:  [1, 4, 2, 5, 3, 6]  ← Community members scattered
Reordered:     [1, 2, 3, 4, 5, 6]  ← Community A then Community B
```

### Algorithm Performance

Graph algorithms typically process vertices and their neighbors:
- **PageRank**: Updates vertex scores based on neighbors
- **BFS**: Explores neighbors in order
- **Triangle Counting**: Checks edges between neighbors

When neighbors are close in memory, these operations are faster.

---

## The Leiden Algorithm

### Overview

Leiden improves upon the popular Louvain algorithm with:
- **Guaranteed connected communities**
- **Better quality** (higher modularity)
- **Faster convergence** in most cases

### How It Works

1. **Move Phase**: Greedily move vertices to improve modularity
2. **Refinement Phase**: Split communities to ensure connectivity
3. **Aggregation Phase**: Create super-graph of communities
4. **Repeat** until no improvement

### Modularity

Modularity measures community quality:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Where:
- $A_{ij}$ = edge between i and j
- $k_i$ = degree of i
- $m$ = total edges
- $\delta(c_i, c_j)$ = 1 if same community

Higher modularity = better community structure.

---

## Leiden in GraphBrew

### Integration

GraphBrew includes a full Leiden implementation in `bench/include/external/leiden/`:

```
leiden/
├── leiden.hxx       # Main Leiden algorithm (LeidenOptions struct)
├── louvain.hxx      # Louvain base implementation
├── Graph.hxx        # Graph representation
├── csr.hxx          # CSR-native operations
├── dfs.hxx          # DFS traversal utilities
├── bfs.hxx          # BFS traversal utilities
├── batch.hxx        # Batch processing support
├── rak.hxx          # Random neighbor sampling
├── properties.hxx   # Graph property computation
└── ...              # Additional utilities
```

### Usage

Leiden is used internally by Leiden-based algorithms:

| Algorithm | Uses Leiden | Format |
|-----------|-------------|--------|
| LeidenOrder (15) | ✓ | `-o 15:resolution` |
| GraphBrewOrder (12) | ✓ | `-o 12:variant:final_algo:resolution` |
| AdaptiveOrder (14) | ✓ | `-o 14:resolution:minsize:mode` |

### Example Output

```
=== Leiden Community Detection ===
Graph: 4039 nodes, 88234 edges
Resolution: 1.0
Iterations: 3
Communities found: 17
Modularity: 0.8347

Community sizes:
  Community 0: 547 nodes
  Community 1: 423 nodes
  Community 2: 398 nodes
  ...
```

---

## Leiden Parameters

### Resolution

Controls community granularity:

| Resolution | Effect | Best For |
|------------|--------|----------|
| < 1.0 | Fewer, larger communities | True community detection |
| 1.0 | Default balance | General use |
| > 1.0 | More, smaller communities | **Cache locality / graph reordering** |

**Key insight:** Higher resolution produces smaller, cache-sized communities → better locality → faster algorithms, despite lower modularity.

See [[AdaptiveOrder-ML#parameters]] for the auto-resolution formula and dynamic/adaptive resolution details.

### Iterations

Defaults from `reorder::ReorderConfig`: `maxIterations=10`, `maxPasses=10`. Usually converges in 2-5 iterations.

---

## Ordering Strategies After Leiden

### LeidenOrder (15)

Hierarchical dendrogram-like ordering:
1. Detect communities using Leiden (multi-pass)
2. Sort vertices by all passes (coarsest to finest)
3. Within same sub-community: sorted by degree (descending)

This achieves RabbitOrder-like locality using Leiden's community structure.

```
Sort key: (pass_N, pass_N-1, ..., pass_0, degree)
Result: Vertices in same sub-sub-community are adjacent
```

### GraphBrewOrder (12)

Fast CSR-native Leiden + per-community reordering. See [[Reordering-Algorithms#12-graphbreworder]] for variant table, resolution modes, and usage examples.

---

## When Leiden Helps Most

### Good Candidates

| Graph Type | Modularity | Speedup |
|------------|------------|---------|
| Social networks | 0.3 - 0.8 | 1.5 - 3x |
| Citation networks | 0.4 - 0.7 | 1.3 - 2x |
| Collaboration | 0.5 - 0.9 | 1.5 - 2.5x |
| Biological | 0.2 - 0.6 | 1.2 - 1.8x |

### Poor Candidates

| Graph Type | Modularity | Speedup |
|------------|------------|---------|
| Road networks | < 0.1 | 1.0 - 1.1x |
| Grids | 0 | 1.0x |
| Random graphs | < 0.05 | 1.0x |

---

## Checking Modularity

### In GraphBrew Output

```
./bench/bin/pr -f graph.el -s -o 14 -n 1

Leiden: 17 communities, modularity 0.8347
```

### Interpreting Modularity

| Range | Interpretation |
|-------|----------------|
| < 0.1 | No community structure |
| 0.1 - 0.3 | Weak structure |
| 0.3 - 0.5 | Moderate structure |
| 0.5 - 0.7 | Strong structure |
| > 0.7 | Very strong structure |

---

## Advanced: Per-Community Analysis

For each community, GraphBrew computes structural features (num_nodes, num_edges, density, avg_degree, degree_variance, hub_concentration, clustering_coeff, modularity). See [[Code-Architecture]] for the `CommunityFeatures` struct in `reorder_types.h`.

---

## Leiden vs Other Methods

| Method | Quality | Speed | Connected Communities |
|--------|---------|-------|-----------------------|
| GVE-Leiden | ★★★★★ | ★★★★ | Yes |
| Louvain (RabbitOrder) | ★★★★ | ★★★★★ | No |
| Infomap | ★★★★★ | ★★★ | No |
| Label Prop | ★★★ | ★★★★★ | No |

**Why Leiden over Louvain?** Leiden's refinement phase prevents bad merges, producing higher modularity with connected communities. Benchmarks show +0.4% to +21% modularity improvement over RabbitOrder's Louvain (e.g., com-Youtube: Q=0.788 vs Q=0.650).

Ref: Traag et al. (2019). *From Louvain to Leiden.* Scientific Reports 9, 5233.

---

## Next Steps

- [[Reordering-Algorithms]] - All reordering techniques
- [[AdaptiveOrder-ML]] - ML-based algorithm selection
- [[Graph-Benchmarks]] - Available benchmarks

---

[← Back to Home](Home) | [Reordering Algorithms →](Reordering-Algorithms)
