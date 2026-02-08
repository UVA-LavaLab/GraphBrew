# Running Benchmarks

Complete guide to running GraphBrew benchmarks with all options explained.

## Overview

The automated pipeline runs **five** benchmarks by default (TC is skipped as it benefits minimally from reordering):

| Benchmark | Binary | Description | Automated |
|-----------|--------|-------------|-----------|
| PageRank | `pr` | Page importance ranking | ✅ |
| BFS | `bfs` | Breadth-First Search traversal | ✅ |
| Connected Components | `cc` | Find graph connectivity | ✅ |
| SSSP | `sssp` | Single-Source Shortest Paths | ✅ |
| Betweenness Centrality | `bc` | Node importance by path flow | ✅ |
| Triangle Counting | `tc` | Count triangles in graph | ❌* |

> **Note:** *Triangle Counting is skipped by default because reordering provides minimal benefit for this workload. To include it, use `--benchmarks pr bfs cc sssp bc tc`.

---

## 🚀 Automated Pipeline (Recommended)

For comprehensive benchmarking, use the unified experiment script:

```bash
# One-click: downloads graphs, builds, runs all benchmarks
python3 scripts/graphbrew_experiment.py --full --size small

# Auto-detect RAM and disk limits
python3 scripts/graphbrew_experiment.py --full --size all --auto

# Specify maximum memory (e.g., 32 GB system)
python3 scripts/graphbrew_experiment.py --full --size all --max-memory 32

# Run benchmarks on existing graphs
python3 scripts/graphbrew_experiment.py --phase benchmark --size small

# Quick test with key algorithms only
python3 scripts/graphbrew_experiment.py --size small --quick

# Use pre-generated label maps for consistent reordering
python3 scripts/graphbrew_experiment.py --precompute --phase benchmark

# Train: complete pipeline (reorder → benchmark → cache sim → weights)
python3 scripts/graphbrew_experiment.py --train --auto --size all
```

See [[Command-Line-Reference]] for `--train` phases, download size options, and memory/disk management. See [[Python-Scripts]] for full script documentation.

---

## Manual Benchmark Execution

### Standard Command Format

```bash
./bench/bin/<benchmark> -f <graph_file> [options]
```

### Minimal Examples

```bash
# PageRank on edge list
./bench/bin/pr -f graph.el -s

# BFS from vertex 0
./bench/bin/bfs -f graph.el -s -r 0

# Triangle counting
./bench/bin/tc -f graph.el -s
```

---

## Command-Line Options

### Input/Output Options

| Option | Description | Default |
|--------|-------------|---------|
| `-f <file>` | Input graph file | Required |
| `-o <id>` | Ordering algorithm (0-17) | 0 (none) |
| `-s` | Symmetrize graph (make undirected) | Off |
| `-g <scale>` | Generate 2^scale kronecker graph | - |
| `-n <num>` | Number of trials | 16 |

### Graph Format Detection

Format is automatically detected from file extension:

| Extension | Format |
|-----------|--------|
| `.el` | Edge list (default) |
| `.wel` | Weighted edge list |
| `.mtx` | Matrix Market |
| `.gr` | DIMACS format |
| `.sg`, `.graph` | Serialized binary |

### Algorithm-Specific Options

| Option | Applicable To | Description |
|--------|---------------|-------------|
| `-r <vertex>` | bfs, sssp, bc | Root/source vertex |
| `-d <delta>` | sssp | Delta for delta-stepping |
| `-i <iter>` | pr | Max iterations |
| `-t <tol>` | pr | Convergence tolerance |

---

## Detailed Benchmark Commands

### PageRank (pr)

```bash
# Basic PageRank
./bench/bin/pr -f graph.el -s -n 5

# With ordering
./bench/bin/pr -f graph.el -s -o 7 -n 5

# Custom iterations and tolerance
./bench/bin/pr -f graph.el -s -i 100 -t 1e-6 -n 5
```

Output:
```
Loading graph from graph.el...
Graph has 4039 nodes and 88234 edges
Trial   Time(s)
1       0.0234
2       0.0231
3       0.0229
Average: 0.0231 seconds
```

### BFS (bfs)

```bash
# BFS from vertex 0
./bench/bin/bfs -f graph.el -s -r 0 -n 5

# BFS from random vertices
./bench/bin/bfs -f graph.el -s -n 5

# With reordering
./bench/bin/bfs -f graph.el -s -o 12 -r 0 -n 5
```

Output:
```
Source: 0
Trial   Time(s)   Edges Visited   MTEPS
1       0.0012    88234           73.5
2       0.0011    88234           80.2
Average: 0.0012 seconds, 76.9 MTEPS
```

### Connected Components (cc)

```bash
# Find connected components
./bench/bin/cc -f graph.el -s -n 5

# With Afforest algorithm variant
./bench/bin/cc -f graph.el -s -o 7 -n 5
```

Output:
```
Largest Component: 4039 (100.00%)
Number of Components: 1
Time: 0.0089 seconds
```

### Single-Source Shortest Paths (sssp)

```bash
# SSSP from vertex 0
./bench/bin/sssp -f graph.wel -s -r 0 -n 5

# With custom delta
./bench/bin/sssp -f graph.wel -s -r 0 -d 2 -n 5

# Using weighted edge list
./bench/bin/sssp -f graph.wel -s -o 7 -n 5
```

Note: SSSP requires weighted edges. Use `.wel` format:
```
0 1 1.5
0 2 2.0
1 2 0.5
```

### Betweenness Centrality (bc)

```bash
# BC from single source
./bench/bin/bc -f graph.el -s -r 0 -n 5

# BC with reordering
./bench/bin/bc -f graph.el -s -o 12 -r 0 -n 5
```

### Triangle Counting (tc)

```bash
# Count triangles
./bench/bin/tc -f graph.el -s -n 5

# With reordering (important for TC!)
./bench/bin/tc -f graph.el -s -o 7 -n 5
```

Output:
```
Number of triangles: 1612010
Time: 1.234 seconds
```

---

## Reordering Options

See [[Command-Line-Reference#reordering-algorithm-ids]] for the full algorithm table (IDs 0-17) and variant syntax.

| Use Case | Algorithm | ID |
|----------|-----------|-----|
| General purpose | HUBCLUSTERDBG | 7 |
| Social networks | LeidenOrder | 15 |
| Unknown graphs | AdaptiveOrder | 14 |
| Maximum locality | LeidenCSR | 17 |
| Road networks | RCM | 11 |

---

## Batch Benchmarking

See [[Command-Line-Reference#common-command-patterns]] for batch scripts (all benchmarks, compare orderings, multiple graphs).

---

## Using Python Scripts

See [[Python-Scripts]] for the full orchestration script reference and [[Command-Line-Reference#python-script-options-graphbrew_experimentpy]] for CLI options.

---

## Output Formats

See [[Command-Line-Reference#output-format]] for standard, BFS, and AdaptiveOrder output examples.

---

## Environment Variables

See [[Command-Line-Reference#environment-variables]] for `OMP_NUM_THREADS`, `PERCEPTRON_WEIGHTS_FILE`, and NUMA binding.

---

## Performance Tips

### For Accurate Timing

1. **Multiple trials**: Always use `-n 5` or more
2. **Warm-up**: First trial may be slower
3. **Disable frequency scaling**: `sudo cpupower frequency-set -g performance`
4. **Dedicated system**: Minimize other processes

### For Large Graphs

1. **Sufficient RAM**: Graph + reordering overhead
2. **Parallel loading**: Already enabled
3. **Try serialized format**: Faster loading for repeated runs

### For Fair Comparisons

1. **Same trials**: Use same `-n` for all runs
2. **Same root**: Use same `-r` for BFS/SSSP/BC
3. **Include reordering time**: Or exclude consistently
4. **Report preprocessing**: Community detection time

---

## Troubleshooting

See [[Troubleshooting]] for solutions to common issues (file not found, invalid format, OOM, segfaults, slow performance).

---

## Next Steps

- [[Graph-Benchmarks]] - Deep dive into each algorithm
- [[Reordering-Algorithms]] - All reordering techniques
- [[AdaptiveOrder-ML]] - ML-powered selection
- [[Supported-Graph-Formats]] - Input format details

---

[← Back to Home](Home) | [Graph Benchmarks →](Graph-Benchmarks)
