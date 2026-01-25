# Graph Benchmarks

GraphBrew includes implementations of classic graph algorithms used to measure the performance impact of vertex reordering. This page explains each algorithm and how to run it.

## Overview of Benchmarks

The automated pipeline uses **five** benchmarks:

| Benchmark | Binary | Description | Complexity |
|-----------|--------|-------------|------------|
| PageRank | `pr` | Importance ranking | O(n + m) per iteration |
| BFS | `bfs` | Shortest paths (unweighted) | O(n + m) |
| Connected Components | `cc` | Find connected subgraphs | O(n + m) |
| SSSP | `sssp` | Shortest paths (weighted) | O((n + m) log n) |
| Betweenness Centrality | `bc` | Node importance | O(n × (n + m)) |

Triangle Counting (`tc`) is also available but excluded from the automated pipeline.

---

## PageRank (pr)

### What is PageRank?

PageRank is an iterative algorithm that computes the "importance" of each vertex based on the importance of vertices pointing to it. Originally developed by Google to rank web pages.

**Mathematical formulation:**
```
PR(v) = (1-d)/n + d × Σ PR(u)/out_degree(u)
        for all u that point to v
```

Where:
- `d` = damping factor (typically 0.85)
- `n` = number of vertices

### How to Run

```bash
# Basic usage
./bench/bin/pr -f graph.el -s -o 0 -n 3

# With generated RMAT graph
./bench/bin/pr -g 20 -o 12 -n 5

# Common options
./bench/bin/pr -f graph.mtx -s -o 17 -n 3 -i 20 -t 1e-6
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-f FILE` | Input graph file | Required |
| `-s` | Graph is symmetric (undirected) | Off |
| `-o N` | Reordering algorithm (0-17) | 0 |
| `-n N` | Number of trials | 16 |
| `-i N` | Max iterations | 20 |
| `-t VAL` | Convergence tolerance | 1e-4 |
| `-g N` | Generate RMAT graph (2^N vertices) | - |

### Output Explained

```
Generate Time:       0.02229    # Time to generate/load graph
Build Time:          0.01802    # Time to build CSR structure
Trial Time:          0.00175    # Time for one PageRank execution
Average Time:        0.00175    # Average over all trials
```

### Why Reordering Helps

PageRank iterates over all vertices multiple times. Each iteration:
1. Reads the current rank of each neighbor
2. Accumulates contributions
3. Writes new rank

**With good ordering**: Neighbors are in cache → fast reads
**With bad ordering**: Neighbors scattered → cache misses → slow

---

## Breadth-First Search (bfs)

### What is BFS?

BFS explores a graph level by level, finding shortest paths (in terms of hops) from a source vertex to all other vertices.

```
Level 0: [source]
Level 1: [neighbors of source]
Level 2: [neighbors of level 1]
...
```

### How to Run

```bash
# Basic usage (3 trials from random sources)
./bench/bin/bfs -f graph.el -s -o 0 -n 3

# More trials for stable timing
./bench/bin/bfs -f graph.el -s -o 17 -n 16

# BFS from specific source vertex
./bench/bin/bfs -f graph.el -s -o 12 -r 0 -n 3

# Generate synthetic graph
./bench/bin/bfs -g 18 -o 12 -n 5
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-f FILE` | Input graph file | Required |
| `-s` | Symmetrize graph (undirected) | Off |
| `-o N` | Reordering algorithm (0-17) | 0 |
| `-n N` | Number of benchmark trials | 16 |
| `-r N` | Starting source vertex | Random |

### Direction-Optimizing BFS

GraphBrew implements **direction-optimizing BFS** which switches between:
- **Top-down**: From frontier to neighbors (when frontier is small)
- **Bottom-up**: Check all unvisited if they connect to frontier (when frontier is large)

This optimization provides ~10-20x speedup on high-diameter graphs!

### Why Reordering Helps

BFS accesses vertices level by level. Good ordering ensures:
- Vertices at the same level are nearby in memory
- Frontier vertices are cached together
- Neighbor lists have spatial locality

---

## Connected Components (cc)

### What is Connected Components?

Finds all maximal sets of vertices where every pair is connected by a path. Returns the component ID for each vertex.

```
Graph: A-B-C  D-E  F
Components: {A,B,C}, {D,E}, {F}
```

### Algorithms

GraphBrew includes two CC algorithms:

1. **Shiloach-Vishkin (cc_sv)**: Parallel with label propagation
2. **Afforest (cc)**: Faster sampling-based approach

### How to Run

```bash
# Standard connected components
./bench/bin/cc -f graph.el -s -o 12 -n 3

# Shiloach-Vishkin variant
./bench/bin/cc_sv -f graph.el -s -o 12 -n 3
```

### Output

```
Components: 1        # Number of connected components found
Largest:    1000000  # Size of largest component
```

---

## Single-Source Shortest Paths (sssp)

### What is SSSP?

Finds the shortest weighted path from a source vertex to all other vertices. Uses Dijkstra's algorithm with a priority queue.

### How to Run

```bash
# Requires weighted graph
./bench/bin/sssp -f weighted_graph.wel -s -o 12 -n 3

# From specific source vertex
./bench/bin/sssp -f graph.wel -s -o 17 -r 0 -n 5

# With custom delta parameter
./bench/bin/sssp -f graph.wel -s -o 12 -d 2 -n 3
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-f FILE` | Input graph (must have weights) | Required |
| `-o N` | Reordering algorithm (0-17) | 0 |
| `-n N` | Number of benchmark trials | 16 |
| `-r N` | Starting source vertex | Random |
| `-d N` | Delta for delta-stepping | 1 |

### Delta-Stepping

GraphBrew uses **delta-stepping SSSP**, which:
- Groups vertices by distance into "buckets"
- Processes buckets in parallel
- Balances work across threads

---

## Betweenness Centrality (bc)

### What is Betweenness Centrality?

Measures how often a vertex lies on the shortest path between other vertices. High BC = vertex is a "bridge" in the network.

**Formula:**
```
BC(v) = Σ σ(s,t|v) / σ(s,t)
        s≠v≠t

Where:
- σ(s,t) = number of shortest paths from s to t
- σ(s,t|v) = number of those paths passing through v
```

### How to Run

```bash
# Basic BC from random source
./bench/bin/bc -f graph.el -s -o 12 -n 3

# BC from specific source vertex
./bench/bin/bc -f graph.el -s -o 12 -r 0 -n 3

# Multiple iterations per trial (more thorough)
./bench/bin/bc -f graph.el -s -o 12 -i 4 -n 3
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-f FILE` | Input graph file | Required |
| `-s` | Symmetrize graph | Off |
| `-o N` | Reordering algorithm (0-17) | 0 |
| `-n N` | Number of benchmark trials | 16 |
| `-r N` | Starting source vertex | Random |
| `-i N` | Number of iterations per trial | 1 |

### Performance Note

BC is computationally expensive: O(n × m) per source vertex. For large graphs:
- Use fewer iterations (`-i 1`) for quick testing
- The `-n` flag controls how many times the benchmark is repeated for timing accuracy
- Each iteration uses a different random source vertex

### Why Reordering Matters Most Here

BC requires running BFS from many source vertices. Each BFS benefits from reordering, so the total benefit compounds!

---

## Triangle Counting (tc)

### What is Triangle Counting?

Counts the number of triangles (3-cliques) in the graph. Important for:
- Social network analysis
- Clustering coefficient computation
- Community detection

```
Triangle: A-B, B-C, C-A
```

### How to Run

```bash
# Standard triangle counting
./bench/bin/tc -f graph.el -s -o 12 -n 3

# Parallel version
./bench/bin/tc_p -f graph.el -s -o 17 -n 3
```

### Algorithms

1. **tc**: Sequential with set intersection
2. **tc_p**: Parallel with work balancing

### Output

```
Triangles: 1234567   # Total number of triangles
```

### Optimization: Degree Ordering

Triangle counting benefits from processing vertices in degree order:
- Low-degree vertices first reduces work
- Combined with reordering for cache efficiency

---

## Running Multiple Benchmarks

### Using the Unified Script (Recommended)

```bash
# One-click: downloads graphs, runs all benchmarks with all algorithms
python3 scripts/graphbrew_experiment.py --full --download-size SMALL

# Run benchmarks on specific graphs
python3 scripts/graphbrew_experiment.py --phase benchmark --graphs small --trials 3

# Run specific benchmarks only
python3 scripts/graphbrew_experiment.py --benchmarks pr bfs --graphs small
```

### Quick Comparison Script

```bash
#!/bin/bash
GRAPH="./results/graphs/email-Enron/email-Enron.mtx"
ALGOS="0 7 12 14 17"  # ORIGINAL, HUBCLUSTERDBG, GraphBrewOrder, AdaptiveOrder, LeidenCSR
TRIALS=3

for algo in $ALGOS; do
    echo "=== Algorithm $algo ==="
    ./bench/bin/pr -f $GRAPH -s -o $algo -n $TRIALS 2>&1 | grep "Average Time"
done
```

---

## Interpreting Results

### What to Compare

1. **Trial Time**: Time for single execution
2. **Average Time**: Mean over multiple trials (more reliable)
3. **Speedup**: `baseline_time / algorithm_time`

### Statistical Significance

- Run at least 3 trials
- For BFS/SSSP, use multiple source vertices (`-n 16`)
- Warm up the cache with one untimed trial first

### Expected Speedups by Benchmark

| Benchmark | Typical Speedup Range |
|-----------|----------------------|
| PageRank | 1.2x - 2.5x |
| BFS | 1.1x - 2.0x |
| CC | 1.1x - 1.5x |
| SSSP | 1.2x - 2.0x |
| BC | 1.3x - 3.0x |
| TC | 1.1x - 1.8x |

---

## Verifying Correctness

### Topology Tests

```bash
# Quick verification
make test-topology

# Full verification with all algorithms
make test-topology-full
```

### Manual Verification

```bash
# Compare outputs between algorithms
./bench/bin/pr -f graph.el -s -o 0 -n 1 > original.txt
./bench/bin/pr -f graph.el -s -o 17 -n 1 > reordered.txt

# Results should match (within floating point tolerance)
```

---

## Next Steps

- [[Running-Benchmarks]] - Detailed command-line reference
- [[Benchmark-Suite]] - Automated benchmarking
- [[Correlation-Analysis]] - Find the best algorithm for your graphs

---

[← Back to Home](Home) | [Reordering Algorithms →](Reordering-Algorithms)
