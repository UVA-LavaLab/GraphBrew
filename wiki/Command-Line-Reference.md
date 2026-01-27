# Command-Line Reference

Complete reference for all GraphBrew command-line options.

---

## Benchmark Binaries

All binaries are located in `bench/bin/`. The automated pipeline uses **six** benchmarks:

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

| Option | Description | Example |
|--------|-------------|---------|
| `-f <file>` | Input graph file (required) | `-f graph.el` |
| `-o <id>` | Reordering algorithm ID (0-17) | `-o 7` |
| `-s` | Make graph undirected (symmetrize) | `-s` |
| `-n <trials>` | Number of benchmark trials | `-n 5` |

### File Format Detection

GraphBrew automatically detects the file format from the file extension:

| Extension | Format | Description |
|-----------|--------|-------------|
| `.el` | Edge list | Text file with "src dst" pairs |
| `.wel` | Weighted edge list | Text file with "src dst weight" |
| `.mtx` | Matrix Market | Standard sparse matrix format |
| `.gr` | DIMACS | DIMACS graph format |
| `.sg` | Serialized graph | Binary format (unweighted) |
| `.wsg` | Weighted serialized | Binary format (weighted) |
| `.graph` | METIS | METIS adjacency format |

**Example:**
```bash
./bench/bin/pr -f graph.el -s -n 5      # Auto-detected as edge list
./bench/bin/sssp -f graph.wel -s -n 5   # Auto-detected as weighted edge list
```

### General Flags

| Option | Description |
|--------|-------------|
| `-h` | Show help message |
| `-v` | Verify results (slower) |
| `-g <scale>` | Generate 2^scale Kronecker graph |
| `-u <scale>` | Generate 2^scale uniform-random graph |
| `-k <degree>` | Average degree for synthetic graph (default: 16) |

---

## Reordering Algorithm IDs

Use with `-o <id>`:

| ID | Algorithm | Category |
|----|-----------|----------|
| 0 | ORIGINAL | None |
| 1 | Random | Basic |
| 2 | Sort | Basic |
| 3 | HubSort | Hub-based |
| 4 | HubCluster | Hub-based |
| 5 | DBG | DBG-based |
| 6 | HubSortDBG | DBG-based |
| 7 | HubClusterDBG | DBG-based |
| 8 | RabbitOrder | Community (variants: csr/boost, default: csr) |
| 9 | GOrder | Community |
| 10 | COrder | Community |
| 11 | RCMOrder | Community |
| 12 | GraphBrewOrder | Community |
| 13 | MAP | External mapping |
| 14 | AdaptiveOrder | ML |
| 15 | LeidenOrder | Leiden (igraph) |
| 16 | LeidenDendrogram | Leiden (variants: dfs/dfshub/dfssize/bfs/hybrid) |
| 17 | LeidenCSR | Leiden (variants: gve/gveopt/dfs/bfs/hubsort/fast/modularity, default: gve) |

---

## Benchmark-Specific Options

### PageRank (pr)

```bash
./bench/bin/pr [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-i <iter>` | Maximum iterations | 20 |
| `-t <tol>` | Convergence tolerance | 1e-4 |

> **Note:** The damping factor is hardcoded to 0.85 in the implementation.

**Examples:**
```bash
# Standard PageRank
./bench/bin/pr -f graph.el -s -n 5

# With custom parameters
./bench/bin/pr -f graph.el -s -i 100 -t 1e-8 -n 5

# With reordering
./bench/bin/pr -f graph.el -s -o 17 -n 5
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
| `-i <iterations>` | Number of source iterations | 1 |

**Examples:**
```bash
# BC from single source
./bench/bin/bc -f graph.el -s -r 0 -n 5

# BC with multiple iterations (more accurate)
./bench/bin/bc -f graph.el -s -i 4 -n 5
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
| `-s` | Symmetrize input |
| `-o <id>` | Apply reordering algorithm |
| `-b <file>` | Output serialized graph (.sg) |
| `-e <file>` | Output edge list (.el) |
| `-p <file>` | Output Matrix Market format (.mtx) |
| `-y <file>` | Output Ligra format (.ligra) |
| `-w` | Make output weighted (.wel/.wsg) |
| `-x <file>` | Output reordered labels as text (.so) |
| `-q <file>` | Output reordered labels as binary (.lo) |

**Examples:**
```bash
# Convert to binary format
./bench/bin/converter -f graph.el -s -b graph.sg

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
# Override default weights file (overrides type matching)
export PERCEPTRON_WEIGHTS_FILE=/path/to/weights.json
./bench/bin/pr -f graph.el -s -o 14 -n 3
```

**Note:** If `PERCEPTRON_WEIGHTS_FILE` is not set, AdaptiveOrder automatically:
1. Computes graph features (modularity, degree variance, hub concentration, etc.)
2. Finds the best matching type file using Euclidean distance to centroids
3. Loads weights from `scripts/weights/active/type_N.json`
4. Falls back to hardcoded defaults if no type files exist

**Example output:**
```
Best matching type: type_0 (distance: 0.4521)
Perceptron: Loaded 5 weights from scripts/weights/active/type_0.json
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
131     1662    16151   0.0117  LeidenCSR
272     103     149     0.0284  Original
...

=== Algorithm Selection Summary ===
Original: 846 communities
LeidenCSR: 3 communities
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
for algo in 0 7 14 15 17; do
    echo "=== Algorithm $algo ==="
    ./bench/bin/pr -f graph.el -s -o $algo -n 3
done
```

### Run All Benchmarks

```bash
for bench in pr bfs cc sssp bc tc; do
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

### Dependency Management

| Option | Description |
|--------|-------------|
| `--check-deps` | Check system dependencies (g++, boost, numa, etc.) |
| `--install-deps` | Install missing system dependencies (requires sudo) |
| `--install-boost` | Download, compile, and install Boost 1.58.0 to /opt/boost_1_58_0 |

### Pipeline Options

| Option | Description |
|--------|-------------|
| `--full` | Run complete pipeline (download → build → experiment → weights) |
| `--download-only` | Only download graphs |
| `--download-size SIZE` | SMALL (16), MEDIUM (28), LARGE (37), XLARGE (6), ALL (87 graphs) |
| `--phase PHASE` | Run specific phase: all, reorder, benchmark, cache, weights, adaptive |

### Memory Management

| Option | Description |
|--------|-------------|
| `--max-memory GB` | Maximum RAM (GB) for graph processing. Graphs exceeding this are skipped. |
| `--auto-memory` | Automatically detect available RAM and skip graphs that won't fit (uses 80% of total) |

Memory estimation: `(edges × 24 bytes + nodes × 8 bytes) × 1.5`

### Disk Space Management

| Option | Description |
|--------|-------------|
| `--max-disk GB` | Maximum disk space (GB) for downloads. Downloads stop when limit is reached. |
| `--auto-disk` | Automatically limit downloads to available disk space (uses 80% of free space) |

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

### Weight Run Management

| Option | Description |
|--------|-------------|
| `--list-runs` | List all saved weight runs in `scripts/weights/runs/` |
| `--merge-runs [TIMESTAMP ...]` | Merge specific runs (or all if no args) into `merged/` |
| `--use-run TIMESTAMP` | Use weights from a specific run (copy to `active/`) |
| `--use-merged` | Use merged weights (copy `merged/` to `active/`) |
| `--no-merge` | Don't auto-merge weights after `--fill-weights` (keep run isolated) |

### Label Map Options

| Option | Description |
|--------|-------------|
| `--generate-maps` | Pre-generate .lo mapping files for consistent reordering |
| `--use-maps` | Use pre-generated label maps instead of regenerating |

### Examples

```bash
# Auto-detect RAM and disk limits, run on all fitting graphs
python3 scripts/graphbrew_experiment.py --full --download-size ALL --auto-memory --auto-disk

# Set explicit 32GB memory and 100GB disk limits
python3 scripts/graphbrew_experiment.py --full --download-size ALL --max-memory 32 --max-disk 100

# Fill all weight fields comprehensively
python3 scripts/graphbrew_experiment.py --fill-weights --auto-memory --auto-disk --download-size ALL

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
4. **Use `-o 14`** (AdaptiveOrder) for unknown graphs
5. **Verify first run** with `-v` to check correctness

---

[← Back to Home](Home) | [Configuration Files →](Configuration-Files)
