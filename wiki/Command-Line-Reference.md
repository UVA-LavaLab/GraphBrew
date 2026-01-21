# Command-Line Reference

Complete reference for all GraphBrew command-line options.

---

## Benchmark Binaries

All binaries are located in `bench/bin/`:

| Binary | Algorithm | Description |
|--------|-----------|-------------|
| `pr` | PageRank | Page importance ranking |
| `bfs` | BFS | Breadth-first search |
| `cc` | Connected Components | Find connected subgraphs |
| `sssp` | Shortest Paths | Single-source shortest paths |
| `bc` | Betweenness Centrality | Centrality measure |
| `tc` | Triangle Counting | Count triangles |
| `converter` | - | Convert graph formats |

---

## Universal Options

These options work with all benchmarks:

### Input/Output

| Option | Long Form | Description | Example |
|--------|-----------|-------------|---------|
| `-f` | `--file` | Input graph file (required) | `-f graph.el` |
| `-o` | `--order` | Reordering algorithm ID (0-20) | `-o 7` |
| `-s` | `--symmetrize` | Make graph undirected | `-s` |
| `-n` | `--num-trials` | Number of benchmark trials | `-n 5` |

### Format Options

| Option | Description | Example |
|--------|-------------|---------|
| `-el` | Force edge list format | `-f data.txt -el` |
| `-wel` | Force weighted edge list | `-f data.txt -wel` |
| `-mtx` | Force Matrix Market format | `-f data.matrix -mtx` |
| `-gr` | Force DIMACS format | `-f data.dimacs -gr` |
| `-sg` | Serialized graph (binary) | `-f data.sg` |

### General Flags

| Option | Long Form | Description |
|--------|-----------|-------------|
| `-h` | `--help` | Show help message |
| `-v` | `--verify` | Verify results (slower) |
| `-g` | `--generate` | Generate synthetic graph |

---

## Reordering Algorithm IDs

Use with `-o <id>`:

| ID | Algorithm | Category |
|----|-----------|----------|
| 0 | ORIGINAL | None |
| 1 | RANDOM | Basic |
| 2 | SORT | Basic |
| 3 | HUBSORT | Hub-based |
| 4 | HUBCLUSTER | Hub-based |
| 5 | DBG | DBG-based |
| 6 | HUBSORTDBG | DBG-based |
| 7 | HUBCLUSTERDBG | DBG-based |
| 8 | RABBITORDER | Community |
| 9 | GORDER | Community |
| 10 | CORDER | Community |
| 11 | RCM | Community |
| 12 | LeidenOrder | Leiden |
| 13 | GraphBrewOrder | Leiden |
| 15 | AdaptiveOrder | ML |
| 16 | LeidenDFS | Leiden |
| 17 | LeidenDFSHub | Leiden |
| 18 | LeidenDFSSize | Leiden |
| 19 | LeidenBFS | Leiden |
| 20 | LeidenHybrid | Leiden |

Note: ID 14 (MAP) is reserved.

---

## Benchmark-Specific Options

### PageRank (pr)

```bash
./bench/bin/pr [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-i <iter>` | Maximum iterations | 20 |
| `-t <tol>` | Convergence tolerance | 1e-6 |
| `-d <damp>` | Damping factor | 0.85 |

**Examples:**
```bash
# Standard PageRank
./bench/bin/pr -f graph.el -s -n 5

# With custom parameters
./bench/bin/pr -f graph.el -s -i 100 -t 1e-8 -n 5

# With reordering
./bench/bin/pr -f graph.el -s -o 20 -n 5
```

### BFS (bfs)

```bash
./bench/bin/bfs [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-r <root>` | Starting vertex | Random |

**Examples:**
```bash
# BFS from vertex 0
./bench/bin/bfs -f graph.el -s -r 0 -n 5

# BFS from random roots
./bench/bin/bfs -f graph.el -s -n 5

# With reordering
./bench/bin/bfs -f graph.el -s -o 7 -r 0 -n 5
```

### Connected Components (cc)

```bash
./bench/bin/cc [options]
```

No additional options.

**Examples:**
```bash
# Find components
./bench/bin/cc -f graph.el -s -n 5

# With verification
./bench/bin/cc -f graph.el -s -v -n 3
```

### SSSP (sssp)

```bash
./bench/bin/sssp [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-r <root>` | Source vertex | 0 |
| `-d <delta>` | Delta for delta-stepping | Auto |

**Examples:**
```bash
# SSSP from vertex 0
./bench/bin/sssp -f graph.wel -s -r 0 -n 5

# With custom delta
./bench/bin/sssp -f graph.wel -s -r 0 -d 2 -n 5
```

**Note:** SSSP requires weighted edges (`.wel` format).

### Betweenness Centrality (bc)

```bash
./bench/bin/bc [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-r <root>` | Source vertex | Random |
| `-k <samples>` | Number of samples | All |

**Examples:**
```bash
# BC from single source
./bench/bin/bc -f graph.el -s -r 0 -n 5

# Approximate BC (faster)
./bench/bin/bc -f graph.el -s -k 64 -n 5
```

### Triangle Counting (tc)

```bash
./bench/bin/tc [options]
```

No additional options.

**Examples:**
```bash
# Count triangles
./bench/bin/tc -f graph.el -s -n 5

# With reordering (important for TC!)
./bench/bin/tc -f graph.el -s -o 7 -n 5
```

### Converter

```bash
./bench/bin/converter [options]
```

| Option | Description |
|--------|-------------|
| `-f <input>` | Input file |
| `-o <output>` | Output file (serialized) |
| `-s` | Symmetrize |

**Examples:**
```bash
# Convert to binary format
./bench/bin/converter -f graph.el -s -o graph.sg

# Use converted graph
./bench/bin/pr -f graph.sg -n 5
```

---

## Environment Variables

### Thread Control

```bash
# Set number of OpenMP threads
export OMP_NUM_THREADS=8
./bench/bin/pr -f graph.el -s -n 5
```

### Perceptron Weights

```bash
# Use custom weights file for AdaptiveOrder
export PERCEPTRON_WEIGHTS_FILE=/path/to/weights.json
./bench/bin/pr -f graph.el -s -o 15 -n 3
```

### NUMA Binding

```bash
# Bind to NUMA node 0
numactl --cpunodebind=0 --membind=0 ./bench/bin/pr -f graph.el -s -n 5
```

---

## Output Format

### Standard Output

```
Loading graph from graph.el...
Graph has 4039 nodes and 88234 edges
Reordering with HUBCLUSTERDBG...

Trial   Time(s)
1       0.0234
2       0.0231
3       0.0229
4       0.0232
5       0.0230

Average: 0.0231 seconds
Std Dev: 0.0002 seconds
```

### With Verification

```
Loading graph from graph.el...
...
Verification: PASSED
```

### BFS Output (includes throughput)

```
Source: 0
Trial   Time(s)   Edges Visited   MTEPS
1       0.0012    88234           73.5
2       0.0011    88234           80.2
3       0.0012    88234           73.5

Average: 0.0012 seconds, 75.7 MTEPS
```

### AdaptiveOrder Output

```
=== Adaptive Reordering Selection ===
Comm    Nodes   Edges   Density Selected
131     1662    16151   0.0117  LeidenHybrid
272     103     149     0.0284  Original
...

=== Algorithm Selection Summary ===
Original: 846 communities
LeidenHybrid: 3 communities
HUBCLUSTERDBG: 2 communities
```

---

## Common Command Patterns

### Quick Test

```bash
./bench/bin/pr -f test/graphs/4.el -s -n 1
```

### Compare Algorithms

```bash
for algo in 0 7 12 15 20; do
    echo "=== Algorithm $algo ==="
    ./bench/bin/pr -f graph.el -s -o $algo -n 3
done
```

### Run All Benchmarks

```bash
for bench in pr bfs cc tc; do
    echo "=== $bench ==="
    ./bench/bin/$bench -f graph.el -s -o 7 -n 3
done
```

### Batch Processing

```bash
for graph in graphs/*.el; do
    echo "=== $graph ==="
    ./bench/bin/pr -f "$graph" -s -o 7 -n 3 | tail -2
done
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| -1 | Argument parsing failed |
| Other | System error |

---

## Python Script Options (graphbrew_experiment.py)

The unified experiment script provides comprehensive options for training and benchmarking.

### Pipeline Options

| Option | Description |
|--------|-------------|
| `--full` | Run complete pipeline (download → build → experiment → weights) |
| `--download-only` | Only download graphs |
| `--download-size SIZE` | SMALL (16), MEDIUM (20), LARGE (20), ALL (56 graphs) |
| `--phase PHASE` | Run specific phase: all, reorder, benchmark, cache, weights, adaptive |

### Training Options

| Option | Description |
|--------|-------------|
| `--train-adaptive` | Run iterative training feedback loop |
| `--train-large` | Large-scale training with batching and multi-benchmark support |
| `--target-accuracy N` | Target accuracy % for training (default: 80) |
| `--max-iterations N` | Maximum training iterations (default: 10) |
| `--learning-rate N` | Weight adjustment rate (default: 0.1) |
| `--fill-weights` | Fill ALL weight fields: runs cache sim, graph features, benchmark analysis |
| `--init-weights` | Initialize/upgrade weights file with enhanced features |

### Label Map Options

| Option | Description |
|--------|-------------|
| `--generate-maps` | Pre-generate .lo mapping files for consistent reordering |
| `--use-maps` | Use pre-generated label maps instead of regenerating |

### Examples

```bash
# Fill all weight fields comprehensively
python3 scripts/graphbrew_experiment.py --fill-weights --graphs small --max-graphs 5

# Iterative training to 85% accuracy
python3 scripts/graphbrew_experiment.py --train-adaptive --target-accuracy 85 --graphs small

# Download and run full pipeline
python3 scripts/graphbrew_experiment.py --full --download-size SMALL
```

See [[Python-Scripts]] for complete documentation.

---

## Tips

1. **Always use `-s`** for undirected graphs
2. **Use `-n 5` or more** for reliable timing
3. **Start with `-o 0`** as baseline
4. **Use `-o 15`** (AdaptiveOrder) for unknown graphs
5. **Verify first run** with `-v` to check correctness

---

[← Back to Home](Home) | [Configuration Files →](Configuration-Files)
