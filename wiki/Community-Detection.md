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

| Resolution | Effect | Best For |
|------------|--------|----------|
| < 1.0 | Fewer, larger communities | True community detection, high modularity |
| 1.0 | Default balance | General use |
| > 1.0 | More, smaller communities | **Cache locality / graph reordering** |

**Key Insight: Resolution Trade-off**

For **graph reordering** (optimizing cache locality for algorithms like PageRank), higher resolution often produces better results:

| Resolution | Modularity | PR Execution | Use Case |
|------------|-----------|--------------|----------|
| 0.3-0.5 | High (0.66-0.74) | Slower | Finding "true" communities |
| 1.0-2.0 | Lower (0.55-0.59) | **Faster** | Cache locality optimization |

This is because:
- Low resolution creates giant communities (high internal density = good modularity)
- But giant communities are too big for cache → poor locality → slow algorithms
- Higher resolution creates balanced, cache-sized communities → faster execution

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

### Dynamic/Adaptive Resolution

The **GVEAdaptive** variant (Algorithm 17:gveadaptive) and **VIBE** (Algorithm 17:vibe:dynamic) dynamically adjust resolution at each Leiden pass based on runtime metrics:

1. **Community reduction rate** - If reducing too fast → raise resolution
2. **Size imbalance** - If giant communities exist → raise resolution to break them
3. **Convergence speed** - If converges in 1 iteration → communities too stable, raise resolution
4. **Super-graph density** - Denser super-graphs need higher resolution

**Algorithms with dynamic resolution support:**

| Algorithm | Syntax | Description |
|-----------|--------|-------------|
| `gveadaptive` | `-o 17:gveadaptive:dynamic` | GVE-Leiden with dynamic |
| `vibe` | `-o 17:vibe:dynamic` | VIBE unified framework |
| `vibe:dfs` | `-o 17:vibe:dfs:dynamic` | VIBE + DFS ordering |
| `vibe:streaming` | `-o 17:vibe:streaming:dynamic` | VIBE + lazy aggregation |
| `vibe:lazyupdate` | `-o 17:vibe:lazyupdate` | VIBE + batched ctot updates |

> **Note:** RabbitOrder variants (`vibe:rabbit`) do not support dynamic resolution and fall back to auto.

Example evolution on wiki-Talk:
```
Pass 0: res=0.500 → 2.4M→64K comms, imbalance=1600x → next_res=1.07
Pass 1: res=1.070 → 64K→4K comms, imbalance=720x  → next_res=1.50
Pass 2: res=1.500 → 4K→3K comms, imbalance=540x   → next_res=2.26
```

### Iterations

Maximum refinement iterations (unified defaults from `reorder::ReorderConfig`):

```cpp
// From reorder_types.h - single source of truth
opts.maxIterations = reorder::DEFAULT_MAX_ITERATIONS;  // 10, usually converges in 2-5
opts.maxPasses = reorder::DEFAULT_MAX_PASSES;          // 10 maximum passes

// Or use unified config directly:
reorder::ReorderConfig cfg = reorder::ReorderConfig::FromOptions(options);
cfg.applyAutoResolution(graph);  // Graph-adaptive resolution
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
2. Apply ordering variant

**Variants (Best to Try First):**

| Variant | Description | Speed | Quality | Recommendation |
|---------|-------------|-------|---------|----------------|
| `gveopt2` | **CSR-based aggregation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Best overall** |
| `gveadaptive` | **Dynamic resolution** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Unknown graphs** |
| `vibe` | **VIBE unified framework** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Configurable** |
| `vibe:dynamic` | **VIBE + per-pass adjustment** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Unknown graphs** |
| `gve` | Standard GVE-Leiden | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Default |
| `gveopt` | Cache-optimized GVE | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Large graphs |
| `gveoptsort` | Multi-level sort | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Hierarchical |
| `gveturbo` | Speed-optimized | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Speed priority |
| `gvefast` | CSR buffer reuse | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Large graphs |
| `gverabbit` | GVE-Rabbit hybrid | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Very large graphs |

**Resolution Modes:**

| Mode | Syntax | Description |
|------|--------|-------------|
| Fixed | `1.5` | Use specified value |
| Auto | `auto` or `0` | Compute from graph density/CV |
| Dynamic | `dynamic` | Auto initial, adjust each pass |
| Dynamic+Init | `dynamic_2.0` | Start at 2.0, adjust each pass |

**Example usage:**
```bash
# Fixed resolution (1.5-2.0 often best for social networks)
./bench/bin/pr -f graph.mtx -s -o 17:gveopt2:2.0 -n 3

# Auto resolution (recommended for unknown graphs)
./bench/bin/pr -f graph.mtx -s -o 17:gveopt2:auto -n 3

# Dynamic resolution (gveadaptive)
./bench/bin/pr -f graph.mtx -s -o 17:gveadaptive:dynamic -n 3

# VIBE with dynamic resolution (recommended unified approach)
./bench/bin/pr -f graph.mtx -s -o 17:vibe:dynamic -n 3

# Dynamic with initial value
./bench/bin/pr -f graph.mtx -s -o 17:gveadaptive:dynamic_2.0 -n 3

# Standard GVE-Leiden with explicit parameters
./bench/bin/pr -f graph.mtx -s -o 17:gve:1.0:20:10 -n 3
```

**Benchmark Results (wiki-Talk: 2.4M nodes):**

| Variant | Reorder Time | PR Execution | vs LeidenOrder |
|---------|--------------|--------------|----------------|
| LeidenOrder (15) | 1.62s | 0.042s | baseline |
| GVEOpt2 res=2.0 | 1.29s | **0.033s** | **21% faster PR** |
| GVEAdaptive res=2.0 | **1.14s** | 0.035s | **30% faster reorder** |

**Comprehensive Benchmark Results (as-Skitter: 1.7M nodes, 11M edges):**

| Variant | Description | Reorder(s) | PR Time(s) | Modularity |
|---------|-------------|------------|------------|------------|
| gve | Standard GVE-Leiden | 3.70 | 0.082 | 0.881 |
| gveopt | Cache-optimized | 1.95 | 0.090 | 0.892 |
| **gveopt2** | **CSR aggregation** | **1.44** | **0.070** | 0.858 |
| gveadaptive | Dynamic resolution | 1.55 | 0.138 | 0.857 |
| gveoptsort | Multi-level sort | 1.72 | 0.095 | 0.893 |
| **gveturbo** | **Speed-optimized** | **0.77** | 0.076 | 0.873 |
| gvefast | CSR buffer reuse | 0.40 | - | 0.768 |

*Takeaway*: `gveopt2` offers the best balance of speed and PR performance. `gveturbo` is fastest for reordering.

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
