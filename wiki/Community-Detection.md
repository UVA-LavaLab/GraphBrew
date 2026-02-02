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
| LeidenDendrogram (16) | ✓ | `-o 16:resolution:variant` |
| LeidenCSR (17) | ✓ | `-o 17:variant:resolution:iterations:passes` |
| GraphBrewOrder (12) | ✓ | `-o 12:freq:algo:resolution` |
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

| Resolution | Effect |
|------------|--------|
| < 1.0 | Fewer, larger communities (better for high-CV graphs) |
| 1.0 | Default balance |
| > 1.0 | More, smaller communities |

**Auto-Resolution Formula:**
```
Base: γ = clip(0.5 + 0.25 × log₁₀(avg_degree + 1), 0.5, 1.2)

CV Adjustment (for power-law graphs with CV > 2.0):
  factor = 2.0 / sqrt(max(2.0, sqrt(cv)))
  γ = max(0.5, γ × factor)
```

**Auto-Resolution by Graph Type:**
| Graph Type | CV | Auto-Resolution |
|------------|-----|----------------|
| Social networks (wiki-Talk) | High (~50) | 0.50 |
| Web graphs | High | 0.50 |
| Email networks | Medium | 0.52 |
| Road networks | Low | 0.60 |
| Co-authorship | Low | 0.77 |

### Iterations

Maximum refinement iterations:

```cpp
opts.maxIterations = 20;  // Default, usually converges in 2-5
opts.maxPasses = 10;      // Maximum number of passes
```

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

### LeidenDendrogram (16)

Dendrogram traversal with variants:
1. Detect communities using igraph Leiden
2. Build community hierarchy
3. Traverse using selected variant (dfs/dfshub/dfssize/bfs/hybrid)

**Variants:**
- `dfs`: Standard DFS traversal
- `dfshub`: DFS with hub-first ordering
- `dfssize`: DFS with size-first ordering  
- `bfs`: BFS level-order traversal
- `hybrid`: Sort by (community, degree) - default

### LeidenCSR (17)

Fast CSR-native Leiden (no graph conversion):
1. Community detection directly on CSR graph
2. Apply ordering variant (gve/gveopt/gverabbit/dfs/bfs/hubsort/fast/modularity)

**Variants:**
- `gve`: GVE-Leiden algorithm (default) - 3-phase: local move, refine, aggregate
- `gveopt`: Cache-optimized GVE with prefetching and flat arrays (faster on large graphs)
- `gverabbit`: GVE-Leiden with RabbitOrder-style intra-community ordering
- `dfs`: Hierarchical DFS
- `bfs`: Level-first BFS
- `hubsort`: Community + degree sort
- `fast`: Union-Find + Label Propagation
- `modularity`: True Leiden with modularity optimization

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

### Community Features (AdaptiveOrder)

For each community, GraphBrew computes:

| Feature | Description |
|---------|-------------|
| `num_nodes` | Community size |
| `num_edges` | Internal edges |
| `internal_density` | edges / possible_edges |
| `avg_degree` | Mean degree |
| `degree_variance` | Normalized variance in degrees |
| `hub_concentration` | Fraction of edges from top 10% nodes |
| `clustering_coeff` | Local clustering coefficient (sampled) |
| `modularity` | Community modularity |

### Feature Extraction Code

```cpp
// From builder.h
struct CommunityFeatures {
    size_t num_nodes;
    size_t num_edges;
    double internal_density;     // edges / possible_edges
    double avg_degree;
    double degree_variance;      // normalized variance in degrees
    double hub_concentration;    // fraction of edges from top 10% nodes
    double modularity;           // community/subgraph modularity
    // Extended features
    double clustering_coeff;
    double avg_path_length;
    double diameter_estimate;
    double community_count;
};
```

---

## Leiden vs Other Methods

### Comparison

| Method | Quality | Speed | Connected |
|--------|---------|-------|-----------|
| GVE-Leiden | ★★★★★ | ★★★★ | Yes |
| Louvain (RabbitOrder) | ★★★★ | ★★★★★ | No |
| Infomap | ★★★★★ | ★★★ | No |
| Label Prop | ★★★ | ★★★★★ | No |

### GVE-Leiden vs RabbitOrder (Louvain) Benchmark

GraphBrew's GVE-Leiden implementation follows the paper: *"Fast Leiden Algorithm for Community Detection in Shared Memory Setting"* (ACM DOI 10.1145/3673038.3673146).

RabbitOrder uses Louvain internally (incremental aggregation, **no refinement phase**).

| Graph | RabbitOrder (Louvain) | GVE-Leiden | Improvement |
|-------|----------------------|------------|-------------|
| web-Google (916K nodes) | 2,959 comm, Q=0.977 | 51,626 comm, Q=0.983 | **+0.6%** |
| roadNet-PA (1.1M nodes) | 443 comm, Q=0.988 | 3,509 comm, Q=0.992 | **+0.4%** |
| com-Youtube (1.1M nodes) | 35 comm, Q=0.650 | 9,168 comm, Q=0.788 | **+21%** |
| Kronecker-18 (262K nodes) | 75 comm, Q=0.063 | 88,227 comm, Q=0.190 | **+3x** |

**Key observation**: GVE-Leiden produces **more communities but higher modularity**. This is because:
- RabbitOrder (Louvain) over-merges small communities into giant poorly-connected "super-communities"
- GVE-Leiden's **refinement phase** prevents bad merges by checking community connectivity
- Smaller, well-connected communities have higher internal density → higher modularity

### Why Leiden over Louvain?

1. **Connected communities**: Louvain can create disconnected groups
2. **Better quality**: Leiden finds higher-modularity partitions (proven in benchmarks above)
3. **Refinement phase**: The key Leiden innovation - only keeps vertices that truly belong together
4. **Avoids resolution limit**: Louvain tends to over-merge, losing fine-grained structure

---

## Troubleshooting

### "Low modularity, no improvement"

Your graph may lack community structure. Try:
- Hub-based algorithms instead (5-7)
- Check if graph is designed for communities

### "Leiden takes too long"

For very large graphs:
- Reduce resolution (larger communities)
- Limit iterations
- Consider approximate methods

### "Wrong number of communities"

Adjust resolution via the algorithm parameter:
```bash
# More communities (higher resolution)
./bench/bin/pr -f graph.el -s -o 17:gve:1.5:20:10 -n 3

# Fewer communities (lower resolution)
./bench/bin/pr -f graph.el -s -o 17:gve:0.5:20:10 -n 3
```

---

## References

- Traag, V.A., Waltman, L., & van Eck, N.J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9, 5233.
- [Leiden Algorithm Wikipedia](https://en.wikipedia.org/wiki/Leiden_algorithm)

---

## Next Steps

- [[Reordering-Algorithms]] - All reordering techniques
- [[AdaptiveOrder-ML]] - ML-based algorithm selection
- [[Graph-Benchmarks]] - Available benchmarks

---

[← Back to Home](Home) | [Reordering Algorithms →](Reordering-Algorithms)
