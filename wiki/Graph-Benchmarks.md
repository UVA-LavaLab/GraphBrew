# Graph Benchmarks

GraphBrew includes implementations of classic graph algorithms used to measure the performance impact of vertex reordering. This page explains each algorithm and how to run it.

## Overview of Benchmarks

The automated pipeline runs **five** benchmarks by default (TC is available but skipped):

| Benchmark | Binary | Description | Complexity |
|-----------|--------|-------------|------------|
| PageRank | `pr` | Importance ranking | O(n + m) per iteration |
| BFS | `bfs` | Shortest paths (unweighted) | O(n + m) |
| Connected Components | `cc` | Find connected subgraphs | O(n + m) |
| SSSP | `sssp` | Shortest paths (weighted) | O((n + m) log n) |
| Betweenness Centrality | `bc` | Node importance | O(n × (n + m)) |

Triangle Counting (`tc`) is available but skipped by default (minimal reorder benefit). Use `--benchmarks pr bfs cc sssp bc tc` to include it.

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
./bench/bin/pr -f graph.el -s -o 16 -n 3
```

See [[Command-Line-Reference]] for all options (`-i` iterations, `-t` tolerance, `-g` synthetic graph).

**Why reordering helps:** PR iterates over all vertices, reading neighbor ranks each iteration. Good ordering = neighbors in cache = fast reads.

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
./bench/bin/bfs -f graph.el -s -o 16 -n 16
./bench/bin/bfs -f graph.el -s -o 12 -r 0 -n 3  # specific source
```

See [[Command-Line-Reference]] for all options. GraphBrew implements **direction-optimizing BFS** (top-down/bottom-up switching) for ~10-20x speedup on high-diameter graphs.

**Why reordering helps:** Vertices at the same BFS level stay nearby in memory → better frontier locality.

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
./bench/bin/sssp -f graph.wel -s -o 12 -n 3
./bench/bin/sssp -f graph.wel -s -o 16 -r 0 -d 2 -n 5  # source 0, delta=2
```

Requires weighted graph (`.wel`). Uses delta-stepping SSSP with parallel bucket processing. See [[Command-Line-Reference]] for options.

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
./bench/bin/bc -f graph.el -s -o 12 -n 3
./bench/bin/bc -f graph.el -s -o 12 -r 0 -i 4 -n 3  # source 0, 4 iters
```

BC is O(n × m) per source vertex. Use `-i 1` for quick testing. Each BFS benefits from reordering, so total benefit **compounds** across iterations. See [[Command-Line-Reference]] for options.

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
./bench/bin/tc_p -f graph.el -s -o 16 -n 3
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

```bash
# One-click: downloads graphs, runs all benchmarks with all algorithms
python3 scripts/graphbrew_experiment.py --full --size small

# Specific benchmarks only
python3 scripts/graphbrew_experiment.py --benchmarks pr bfs --size small
```

See [[Benchmark-Suite]] for automated experiments and [[Running-Benchmarks]] for manual usage.

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
./bench/bin/pr -f graph.el -s -o 16 -n 1 > reordered.txt

# Results should match (within floating point tolerance)
```

---

## Next Steps

- [[Running-Benchmarks]] - Detailed command-line reference
- [[Benchmark-Suite]] - Automated benchmarking
- [[Correlation-Analysis]] - Find the best algorithm for your graphs

---

[← Back to Home](Home) | [Reordering Algorithms →](Reordering-Algorithms)
