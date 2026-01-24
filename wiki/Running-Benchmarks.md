# Running Benchmarks

Complete guide to running GraphBrew benchmarks with all options explained.

## Overview

GraphBrew includes six graph algorithm benchmark binaries from the GAP Benchmark Suite. The automated pipeline uses **five** benchmarks (PR, BFS, CC, SSSP, BC), while Triangle Counting (TC) is available for manual use:

| Benchmark | Binary | Description | Automated |
|-----------|--------|-------------|-----------|
| PageRank | `pr` | Page importance ranking | ✅ |
| BFS | `bfs` | Breadth-First Search traversal | ✅ |
| Connected Components | `cc` | Find graph connectivity | ✅ |
| SSSP | `sssp` | Single-Source Shortest Paths | ✅ |
| Betweenness Centrality | `bc` | Node importance by path flow | ✅ |
| Triangle Counting | `tc` | Count triangles in graph | ❌* |

> **Note:** *Triangle Counting is excluded from the automated pipeline because reordering provides minimal benefit for this workload. It can still be run manually.

---

## 🚀 Automated Pipeline (Recommended)

For comprehensive benchmarking, use the unified experiment script:

```bash
# One-click: downloads graphs, builds, runs all benchmarks
python3 scripts/graphbrew_experiment.py --full --download-size SMALL

# Auto-detect RAM and skip graphs that won't fit
python3 scripts/graphbrew_experiment.py --full --download-size ALL --auto-memory

# Specify maximum memory (e.g., 32 GB system)
python3 scripts/graphbrew_experiment.py --full --download-size ALL --max-memory 32

# Run benchmarks on existing graphs
python3 scripts/graphbrew_experiment.py --phase benchmark --graphs small

# Quick test with key algorithms only
python3 scripts/graphbrew_experiment.py --graphs small --key-only

# Use pre-generated label maps for consistent reordering
python3 scripts/graphbrew_experiment.py --generate-maps --use-maps --phase benchmark

# Fill ALL weight fields (graph type detection, cache sim, topology features, per-type weights)
python3 scripts/graphbrew_experiment.py --fill-weights --auto-memory --download-size ALL
```

### `--fill-weights` Phases

The comprehensive `--fill-weights` mode runs these phases:

| Phase | Description |
|-------|-------------|
| **Phase 0** | Graph Property Analysis - computes modularity, degree variance, hub concentration, detects graph type |
| **Phase 1** | Generate Reorderings - tests all algorithms, records reorder times |
| **Phase 2** | Execution Benchmarks - runs PR, BFS, CC, SSSP, BC with all algorithms |
| **Phase 3** | Cache Simulation - measures L1/L2/L3 hit rates |
| **Phase 4** | Generate Base Weights - creates initial perceptron weights |
| **Phase 5** | Update Topology Weights - fills clustering coefficient, path length, etc. |
| **Phase 6** | Compute Benchmark Weights - per-benchmark multipliers |
| **Phase 7** | Auto-Cluster Type Weights - clusters graphs and creates type_N.json files |

Output files are saved to `scripts/weights/`: `type_0.json`, `type_1.json`, `type_registry.json`, etc.

### Download Options

| Size | Graphs | Total | Categories |
|------|--------|-------|------------|
| `SMALL` | 16 | ~62 MB | communication, p2p, social |
| `MEDIUM` | 28 | ~1.1 GB | web, road, commerce, mesh, synthetic |
| `LARGE` | 37 | ~25 GB | social, web, collaboration, road |
| `XLARGE` | 6 | ~63 GB | massive web (twitter7, webbase), Kronecker |
| `ALL` | **87** | ~89 GB | Complete benchmark set |

### Memory Management

The script automatically estimates memory requirements for each graph:

```
Memory ≈ (edges × 24 bytes + nodes × 8 bytes) × 1.5 safety factor
```

| Option | Description |
|--------|-------------|
| `--max-memory GB` | Skip graphs requiring more than this RAM |
| `--auto-memory` | Use 80% of total system RAM as limit |

Example memory requirements:
| Graph | Nodes | Edges | Memory Required |
|-------|-------|-------|-----------------|
| email-Enron | 37K | 184K | ~0.01 GB |
| soc-LiveJournal1 | 4.8M | 69M | ~2.4 GB |
| com-Orkut | 3.1M | 117M | ~4.0 GB |
| uk-2005 | 39M | 936M | ~32 GB |
| twitter7 | 42M | 1.5B | ~52 GB |

### Disk Space Management

The script can limit downloads based on available disk space:

| Option | Description |
|--------|-------------|
| `--max-disk GB` | Stop downloads when cumulative size exceeds this limit |
| `--auto-disk` | Use 80% of available disk space as limit |

See [[Python-Scripts]] for full documentation.

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
| `-o <id>` | Ordering algorithm (0-20) | 0 (none) |
| `-s` | Symmetrize graph | Off |
| `-g` | Make graph undirected | Off |
| `-n <num>` | Number of trials | 1 |

### Graph Format Options

| Option | Description |
|--------|-------------|
| `-el` | Edge list format (default) |
| `-mtx` | Matrix Market format |
| `-gr` | DIMACS format |
| `-sg` | Serialized graph |

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

### All Available Algorithms

```bash
# No reordering
-o 0   # ORIGINAL

# Basic reordering
-o 1   # RANDOM
-o 2   # SORT (by degree)

# Hub-based
-o 3   # HUBSORT
-o 4   # HUBCLUSTER
-o 5   # DBG
-o 6   # HUBSORTDBG
-o 7   # HUBCLUSTERDBG

# Community-based
-o 8   # RABBITORDER
-o 9   # GORDER
-o 10  # CORDER
-o 11  # RCM

# Leiden-based
-o 12  # LeidenOrder
-o 13  # GraphBrewOrder
-o 14  # AdaptiveOrder (ML)
-o 14  # LeidenDFS
-o 16  # LeidenDFSHub
-o 17  # LeidenDFSSize
-o 19  # LeidenBFS
-o 20  # LeidenHybrid
```

### Recommended Algorithms by Use Case

| Use Case | Algorithm | ID |
|----------|-----------|-----|
| General purpose | HUBCLUSTERDBG | 7 |
| Social networks | LeidenOrder | 12 |
| Unknown graphs | AdaptiveOrder (14) |
| Maximum locality | LeidenHybrid | 20 |
| Road networks | RCM | 11 |
| Quick test | ORIGINAL | 0 |

---

## Batch Benchmarking

### Run All Benchmarks on One Graph

```bash
#!/bin/bash
GRAPH=my_graph.el
ORDER=7

for bench in pr bfs cc tc; do
    echo "=== $bench ==="
    ./bench/bin/$bench -f $GRAPH -s -o $ORDER -n 5
done
```

### Compare Multiple Orderings

```bash
#!/bin/bash
GRAPH=my_graph.el

for order in 0 7 12 15 20; do
    echo "=== Order $order ==="
    ./bench/bin/pr -f $GRAPH -s -o $order -n 3
done
```

### Run on Multiple Graphs

```bash
#!/bin/bash
for graph in graphs/*.el; do
    echo "=== $graph ==="
    ./bench/bin/pr -f "$graph" -s -o 7 -n 3
done
```

---

## Using Python Scripts

### graph_brew.py - Full Pipeline

```bash
cd scripts
python3 graph_brew.py \
    --input ../graphs/my_graph.el \
    --benchmark pr bfs \
    --algorithms 0 7 12 20 \
    --trials 5 \
    --output results.csv
```

### run_experiment.py - Experiment Runner

```bash
cd scripts/brew
python3 run_experiment.py \
    --config ../config/brew/run.json \
    --graphs-dir ../graphs
```

---

## Output Formats

### Standard Output

```
Loading graph from graph.el...
Reordering with HUBCLUSTERDBG...
Graph has 4039 nodes and 88234 edges

Trial   Time(s)
1       0.0234
2       0.0231
3       0.0229
4       0.0232
5       0.0230

Average: 0.0231 seconds
Std Dev: 0.0002 seconds
```

### Timing Breakdown (Verbose)

With reordering, additional timing info:
```
=== Timing Breakdown ===
Graph loading:     0.012s
Community detection: 0.045s
Reordering:        0.023s
Benchmark (avg):   0.0231s
Total:             0.103s
```

### CSV Output (with scripts)

```csv
graph,algorithm,order,trial,time,metric
facebook.el,pr,7,1,0.0234,
facebook.el,pr,7,2,0.0231,
facebook.el,bfs,7,1,0.0012,76.9
```

---

## Environment Variables

### Perceptron Weights

```bash
# Use custom weights file
export PERCEPTRON_WEIGHTS_FILE=/path/to/weights.json
./bench/bin/pr -f graph.el -s -o 14 -n 3
```

### OpenMP Threads

```bash
# Control parallelism
export OMP_NUM_THREADS=8
./bench/bin/pr -f graph.el -s -o 7 -n 3
```

### Numa Binding

```bash
# Bind to specific NUMA node
numactl --cpunodebind=0 --membind=0 ./bench/bin/pr -f graph.el -s -n 5
```

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

### "File not found"

```bash
# Check file exists
ls -la graph.el

# Use absolute path
./bench/bin/pr -f /full/path/to/graph.el -s
```

### "Invalid format"

```bash
# Check file format
head -5 graph.el
# Should be: node1 node2 [weight]

# Remove headers if present
tail -n +2 graph.el > graph_clean.el
```

### "Out of memory"

```bash
# Check available memory
free -h

# Try smaller graph or more RAM
```

### "Segmentation fault"

```bash
# Build with debug
make clean
make DEBUG=1

# Run with debugger
gdb ./bench/bin/pr
```

### Slow Performance

```bash
# Check thread count
echo $OMP_NUM_THREADS

# Use all cores
export OMP_NUM_THREADS=$(nproc)
```

---

## Next Steps

- [[Graph-Benchmarks]] - Deep dive into each algorithm
- [[Reordering-Algorithms]] - All reordering techniques
- [[AdaptiveOrder-ML]] - ML-powered selection
- [[Supported-Graph-Formats]] - Input format details

---

[← Back to Home](Home) | [Graph Benchmarks →](Graph-Benchmarks)
