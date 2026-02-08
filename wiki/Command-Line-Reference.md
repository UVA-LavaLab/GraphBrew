# Command-Line Reference

Complete reference for all GraphBrew command-line options.

---

## Quick Reference: Evaluation vs Training

### Evaluation Modes (No Weight Updates)

| Mode | Command | Description |
|------|---------|-------------|
| **Reorder Only** | `--phase reorder --size small` | Test reordering algorithms only |
| **Benchmark Only** | `--phase benchmark --size small --skip-cache` | Run graph algorithm benchmarks (BFS, PR, etc.) |
| **End-to-End** | `--full --size small --auto` | Full evaluation pipeline without training |
| **Validation** | `--brute-force --validation-benchmark pr` | Compare AdaptiveOrder vs all algorithms |

### Training Modes (Updates Weights)

| Mode | Command | Description |
|------|---------|-------------|
| **Standard** | `--train --size small --auto` | One-pass training: reorder → benchmark → cache → weights |
| **Iterative** | `--train-iterative --target-accuracy 90` | Repeated training until target accuracy |
| **Batched** | `--train-batched --size medium --batch-size 8` | Large-scale batched training |

### Common Modifiers

| Modifier | Description |
|----------|-------------|
| `--quick` | Key algorithms only (faster) |
| `--skip-cache` | Skip cache simulation (faster) |
| `--skip-expensive` | Skip BC/SSSP benchmarks |
| `--all-variants` | Test all algorithm variants |
| `--auto` | Auto-detect memory/disk limits |
| `--precompute` | Use pre-generated label maps |

### Run Phases Separately

You can run each phase independently. Later phases automatically load results from earlier phases:

| Phase | Command | Description |
|-------|---------|-------------|
| **Phase 1** | `--phase reorder` | Generate reordered graphs (.lo label maps) |
| **Phase 2** | `--phase benchmark` | Run graph algorithm benchmarks (BFS, PR, etc.) |
| **Phase 3** | `--phase cache` | Run cache simulation |
| **Phase 4** | `--phase weights` | Generate perceptron weights from results |

```bash
# Run each phase separately
python3 scripts/graphbrew_experiment.py --phase reorder --size small
python3 scripts/graphbrew_experiment.py --phase benchmark --size small
python3 scripts/graphbrew_experiment.py --phase cache --size small
python3 scripts/graphbrew_experiment.py --phase weights

# Or chain them
python3 scripts/graphbrew_experiment.py --phase reorder --size small && \
python3 scripts/graphbrew_experiment.py --phase benchmark --size small && \
python3 scripts/graphbrew_experiment.py --phase cache --size small && \
python3 scripts/graphbrew_experiment.py --phase weights
```

**Note:** Results are saved to `results/` directory after each phase. Later phases automatically load:
- Phase 2 & 3: Load `.lo` label maps from Phase 1
- Phase 4: Load `benchmark_*.json`, `cache_*.json`, `reorder_*.json` from Phases 1-3

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
| `-j type:n:m` | Partition graph (`type=0` Cagra, `1` TRUST), default `0:1:1` | `-j 0:2:2` |
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

### Partitioning / Segmentation

| Type | Partitioning | Implementation | Notes |
|------|--------------|----------------|-------|
| `0` | **Cagra/GraphIT** CSR slicing | `cache/popt.h` → `MakeCagraPartitionedGraph` | Uses `graphSlicer`, honors `-z` (use out-degree) |
| `1` | **TRUST** (triangle counting) | `partition/trust.h` → `TrustPartitioner::MakeTrustPartitionedGraph` | Orients edges, partitions p_n × p_m |

> **Tip:** Cache **simulation** headers live in `bench/include/cache_sim/` (`cache_sim.h`, `graph_sim.h`). Cagra partition helpers live in `bench/include/graphbrew/partition/cagra/` (`popt.h`). See `docs/INDEX.md` for a quick map.

Examples:
```bash
# Cagra partitioning into 2x2 segments (out-degree)
./bench/bin/pr -f graph.mtx -j 0:2:2

# TRUST partitioning into 2x2 segments
./bench/bin/tc -f graph.mtx -j 1:2:2
```

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
| 8 | RabbitOrder | Community (has variants, see below) |
| 9 | GOrder | Community |
| 10 | COrder | Community |
| 11 | RCMOrder | Community |
| 12 | GraphBrewOrder | Community (has variants, see below) |
| 13 | MAP | External mapping |
| 14 | AdaptiveOrder | ML |
| 15 | LeidenOrder | Leiden (igraph) |
| 16 | LeidenDendrogram | Leiden (has variants, see below) |
| 17 | LeidenCSR | Leiden (has variants, including VIBE) |

> **Note:** For current variant lists, see `scripts/lib/utils.py` which defines:
> - `RABBITORDER_VARIANTS`, `GRAPHBREW_VARIANTS`
> - `LEIDEN_DENDROGRAM_VARIANTS`, `LEIDEN_CSR_VARIANTS`
> - `VIBE_LEIDEN_VARIANTS`, `VIBE_RABBIT_VARIANTS`

### LeidenCSR Resolution Modes (Algorithm 17)

| Mode | Syntax | Example | Description |
|------|--------|---------|-------------|
| **Fixed** | `<value>` | `-o 17:gveopt2:1.5` | Use specified resolution |
| **Auto** | `auto` or `0` | `-o 17:gveopt2:auto` | Compute from graph density |
| **Dynamic** | `dynamic` | `-o 17:gveadaptive:dynamic` | Auto initial, adjust per-pass |
| **Dynamic+Init** | `dynamic_<val>` | `-o 17:gveadaptive:dynamic_2.0` | Start at value, adjust per-pass |

### VIBE Variants (Algorithm 17)

VIBE (Vertex Indexing for Better Efficiency) provides a unified reordering framework with two algorithms:

**Leiden-based VIBE** (multi-pass community detection):

| Variant | Example | Description |
|---------|---------|-------------|
| `vibe` | `-o 17:vibe` | Hierarchical ordering (default) |
| `vibe:dfs` | `-o 17:vibe:dfs` | DFS dendrogram traversal |
| `vibe:bfs` | `-o 17:vibe:bfs` | BFS dendrogram traversal |
| `vibe:dbg` | `-o 17:vibe:dbg` | DBG within each community |
| `vibe:corder` | `-o 17:vibe:corder` | Hot/cold within communities |
| `vibe:dbg-global` | `-o 17:vibe:dbg-global` | DBG across all vertices |
| `vibe:corder-global` | `-o 17:vibe:corder-global` | Hot/cold across all vertices |
| `vibe:streaming` | `-o 17:vibe:streaming` | Leiden + lazy aggregation |
| `vibe:lazyupdate` | `-o 17:vibe:lazyupdate` | Batched community weight updates (reduces atomics) |
| `vibe:conn` | `-o 17:vibe:conn` | Connectivity BFS within communities (Boost-style, default ordering) |
| `vibe:hrab` | `-o 17:vibe:hrab` | Hybrid Leiden+RabbitOrder (best locality on web/geometric graphs) |
| `vibe:auto` | `-o 17:vibe:auto` | Auto-computed resolution (computed once) |
| `vibe:dynamic` | `-o 17:vibe:dynamic` | Dynamic resolution (adjusted per-pass) |

**VIBE Resolution Modes:**

| Mode | Syntax | Description |
|------|--------|-------------|
| Auto | `-o 17:vibe:auto` or `-o 17:vibe` | Compute resolution from graph density/CV (fixed) |
| Dynamic | `-o 17:vibe:dynamic` | Auto initial, adjust each pass based on metrics |
| Fixed | `-o 17:vibe:1.5` | Use specified resolution value |
| Dynamic+Init | `-o 17:vibe:dynamic_2.0` | Start at 2.0, adjust each pass |

> **Note:** Dynamic resolution adjusts based on community reduction rate, size imbalance, and convergence speed. Use for unknown graphs.

**RabbitOrder-based VIBE** (single-pass parallel aggregation):

| Variant | Example | Description |
|---------|---------|-------------|
| `vibe:rabbit` | `-o 17:vibe:rabbit` | RabbitOrder algorithm (DFS default) |
| `vibe:rabbit:dfs` | `-o 17:vibe:rabbit:dfs` | RabbitOrder + DFS post-ordering |
| `vibe:rabbit:bfs` | `-o 17:vibe:rabbit:bfs` | RabbitOrder + BFS post-ordering |
| `vibe:rabbit:dbg` | `-o 17:vibe:rabbit:dbg` | RabbitOrder + DBG post-ordering |
| `vibe:rabbit:corder` | `-o 17:vibe:rabbit:corder` | RabbitOrder + COrder post-ordering |

> **Note:** RabbitOrder variants do not support dynamic resolution (falls back to auto).

### GraphBrewOrder Variants (Algorithm 12)

| Variant | Description | Resolution | Final Algo |
|---------|-------------|------------|------------|
| `leiden` | GVE-Leiden optimized (default) | auto | RabbitOrder |
| `gve` | GVE-Leiden non-optimized | auto | RabbitOrder |
| `gveopt` | GVE-Leiden with cache optimization | auto | RabbitOrder |
| `gvefast` | GVE-Leiden non-optimized | auto | HubSortDBG |
| `gveoptfast` | GVE-Leiden optimized | auto | HubSortDBG |
| `rabbit` | GVE-Leiden with coarse communities | 0.50 | RabbitOrder |
| `hubcluster` | Hub-degree based clustering | N/A | RabbitOrder |

**Auto-Resolution:** Automatically computed based on graph's coefficient of variation (CV):
- High-CV graphs (social/web): resolution ≈ 0.50 (coarser communities, better locality)
- Low-CV graphs (road networks): resolution ≈ 0.60-0.77 (finer communities)

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

**Note:** If not set, AdaptiveOrder uses type-based weight loading (see [[Perceptron-Weights#weight-loading-priority]]).

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
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -n 1
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

See [[Python-Scripts]] for complete script documentation and module reference.

**Key options summary:**

| Category | Key Options |
|----------|-------------|
| Pipeline | `--full`, `--train`, `--train-iterative`, `--train-batched`, `--phase PHASE` |
| Size/Resources | `--size small\|medium\|large`, `--auto`, `--max-memory GB` |
| Speed | `--quick`, `--skip-cache`, `--skip-expensive`, `--skip-slow` |
| Variants | `--all-variants`, `--csr-variants LIST`, `--resolution VALUE` |
| Weights | `--isolate-run`, `--merge-runs`, `--use-run TIMESTAMP` |
| Labels | `--precompute`, `--generate-maps`, `--use-maps` |
| Validation | `--brute-force`, `--validation-benchmark NAME` |
| Dependencies | `--check-deps`, `--install-deps`, `--install-boost` |

```bash
# Quick examples
python3 scripts/graphbrew_experiment.py --full --size small --auto --quick
python3 scripts/graphbrew_experiment.py --train --size medium --auto --precompute
python3 scripts/graphbrew_experiment.py --brute-force --validation-benchmark bfs
```

### eval_weights.py

Trains weights and simulates C++ scoring to report accuracy/regret. See [[Python-Scripts#-eval_weightspy---weight-evaluation--c-scoring-simulation]].

```bash
python3 scripts/eval_weights.py  # No arguments needed
```

---

## Tips

1. **Always use `-s`** for undirected graphs
2. **Use `-n 5` or more** for reliable timing
3. **Start with `-o 0`** as baseline
4. **Use `-o 14`** (AdaptiveOrder) for unknown graphs
5. **Verify first run** with `-v` to check correctness

---

[← Back to Home](Home) | [Configuration Files →](Configuration-Files)
