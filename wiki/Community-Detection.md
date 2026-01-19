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

GraphBrew includes a full Leiden implementation in `bench/include/leiden/`:

```
leiden/
├── leiden.hxx       # Main algorithm
├── louvain.hxx      # Louvain base
├── Graph.hxx        # Graph representation
├── properties.hxx   # Community properties
└── ...
```

### Usage

Leiden is used internally by Leiden-based algorithms:

| Algorithm | Uses Leiden |
|-----------|-------------|
| LeidenOrder (12) | ✓ |
| GraphBrewOrder (13) | ✓ |
| AdaptiveOrder (15) | ✓ |
| LeidenDFS (16) | ✓ |
| LeidenDFSHub (17) | ✓ |
| LeidenDFSSize (18) | ✓ |
| LeidenBFS (19) | ✓ |
| LeidenHybrid (20) | ✓ |

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
| < 1.0 | Fewer, larger communities |
| 1.0 | Default balance |
| > 1.0 | More, smaller communities |

```cpp
// In builder.h
LeidenOptions opts;
opts.resolution = 1.0;  // Default
```

### Iterations

Maximum refinement iterations:

```cpp
opts.max_iterations = 10;  // Usually converges in 2-5
```

---

## Ordering Strategies After Leiden

### LeidenOrder (12)

Simple contiguous ordering:
1. Detect communities
2. Assign IDs by community order
3. Within community: original order

```
Community 0: vertices get IDs 0, 1, 2, ...
Community 1: vertices get IDs n0, n0+1, ...
```

### LeidenDFS (16)

DFS traversal within communities:
1. Detect communities
2. For each community: DFS from highest-degree vertex
3. Concatenate communities

**Benefits**: Better locality for tree-like structures

### LeidenBFS (19)

BFS traversal within communities:
1. Detect communities
2. For each community: BFS from highest-degree vertex
3. Concatenate communities

**Benefits**: Better for broad, shallow structures

### LeidenHybrid (20)

Combines multiple strategies:
1. Detect communities
2. Try DFS, BFS, and hub-based ordering
3. Select best based on estimated locality

**Benefits**: Adapts to community structure

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
./bench/bin/pr -f graph.el -s -o 12 -n 1

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
| `density` | edges / max_edges |
| `avg_degree` | Mean degree |
| `degree_variance` | Degree spread |
| `hub_concentration` | Fraction to top 10% |

### Feature Extraction Code

```cpp
// Simplified from builder.h
struct CommunityFeatures {
    int64_t num_nodes;
    int64_t num_edges;
    double density;
    double avg_degree;
    double degree_variance;
    double hub_concentration;
};

CommunityFeatures ComputeFeatures(const Community& c) {
    // ... compute statistics
}
```

---

## Leiden vs Other Methods

### Comparison

| Method | Quality | Speed | Connected |
|--------|---------|-------|-----------|
| Leiden | ★★★★★ | ★★★★ | Yes |
| Louvain | ★★★★ | ★★★★★ | No |
| Infomap | ★★★★★ | ★★★ | No |
| Label Prop | ★★★ | ★★★★★ | No |

### Why Leiden over Louvain?

1. **Connected communities**: Louvain can create disconnected groups
2. **Better quality**: Leiden finds higher-modularity partitions
3. **Consistent results**: Less variance across runs

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

Adjust resolution:
```bash
# More communities
export LEIDEN_RESOLUTION=1.5

# Fewer communities  
export LEIDEN_RESOLUTION=0.5
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
