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
| `--pregenerate-sg` | Pre-generate reordered `.sg` per algorithm (default ON) |
| `--no-pregenerate-sg` | Disable `.sg` pre-generation; reorder at runtime instead |

### Run Phases Separately

You can run each phase independently. Later phases automatically load results from earlier phases:

| Phase | Command | Description |
|-------|---------|-------------|
| **Phase 0** | (automatic) | Convert `.mtx` → `.sg` with RANDOM baseline + pre-generate reordered `.sg` per algorithm |
| **Phase 1** | `--phase reorder` | Generate reordered graphs (.lo label maps) |
| **Phase 2** | `--phase benchmark` | Run graph algorithm benchmarks (BFS, PR, etc.) |
| **Phase 3** | `--phase cache` | Run cache simulation |
| **Phase 4** | `--phase weights` | _(deprecated)_ Generate perceptron weights — C++ now trains at runtime |
| **Phase 5** | `--phase adaptive` | _(deprecated)_ Adaptive order analysis — use `--eval-weights` instead |

```bash
# Run each phase separately
python3 scripts/graphbrew_experiment.py --phase reorder --size small
python3 scripts/graphbrew_experiment.py --phase benchmark --size small
python3 scripts/graphbrew_experiment.py --phase cache --size small

# Or chain them
python3 scripts/graphbrew_experiment.py --phase reorder --size small && \
python3 scripts/graphbrew_experiment.py --phase benchmark --size small && \
python3 scripts/graphbrew_experiment.py --phase cache --size small
```

**Note:** Results are saved to `results/` directory after each phase. Later phases automatically load:
- Phase 2 & 3: Load `.lo` label maps from Phase 1
- C++ runtime: Trains perceptron, decision tree, and hybrid models automatically from benchmark data when ≥3 graphs are available

---

## Benchmark Binaries

All binaries are located in `bench/bin/`. The automated pipeline uses **seven** benchmarks by default (TC excluded from experiments):

| Binary | Algorithm | Description |
|--------|-----------|-------------|
| `pr` | PageRank (pull) | Page importance ranking |
| `pr_spmv` | PageRank (SpMV) | Sparse matrix-vector PageRank |
| `bfs` | BFS | Breadth-first search |
| `cc` | Connected Components (Afforest) | Find connected subgraphs |
| `cc_sv` | Connected Components (SV) | Shiloach-Vishkin CC |
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
| `-o <id>` | Reordering algorithm ID (0-15) | `-o 7` |
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
| `-m` | Reduce memory usage during graph building |
| `-l` | Log performance within each trial |
| `-a` | Output analysis of last run |
| `-S` | Keep self-loops (default: removed) |

### Partitioning / Segmentation

| Type | Partitioning | Implementation | Notes |
|------|--------------|----------------|-------|
| `0` | **Cagra/GraphIT** CSR slicing | `cache/popt.h` → `MakeCagraPartitionedGraph` | Uses `graphSlicer`, honors `-z` (use indegree) |
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
| 9 | GOrder | Cache-based (has variants, see below) |
| 10 | COrder | Cache-based |
| 11 | RCMOrder | Classic (has variants, see below) |
| 12 | GraphBrewOrder | Community (has variants, see below) |
| 13 | MAP | External mapping |
| 14 | AdaptiveOrder | ML |
| 15 | LeidenOrder | Leiden (GVE-Leiden baseline) |

> **Note:** For current variant lists, see `scripts/lib/core/utils.py` which defines:
> - `RABBITORDER_VARIANTS`, `GORDER_VARIANTS`, `RCM_VARIANTS`, `GRAPHBREW_VARIANTS`
> - Use `get_algo_variants(algo_id)` to query programmatically

### GOrder Variants (Algorithm 9)

GOrder supports three variants:

| Variant | Example | Description |
|---------|---------|-------------|
| (default) | `-o 9` | GoGraph baseline — converts to GoGraph adjacency format |
| `csr` | `-o 9:csr` | CSR-native — direct CSRGraph iterator access, lightweight BFS-RCM, faster reorder |
| `fast` | `-o 9:fast` | Parallel batch — atomic score updates, fan-out cap, scales across threads |

The CSR variant uses a lightweight GoGraph-matching BFS-CM pre-ordering and `RelabelByMappingStandalone` to rebuild the CSR in RCM order, then runs the GOrder greedy directly on sorted CSRGraph neighbor iterators. Deterministic with single thread.

The fast variant replaces the serial UnitHeap with a score array + active frontier for thread safety. It auto-tunes batch size and window to the available thread count. Recommended for graphs with high degree variance (power-law, social networks).

### RCMOrder Variants (Algorithm 11)

RCM supports two variants:

| Variant | Example | Description |
|---------|---------|-------------|
| (default) | `-o 11` | GoGraph double-RCM (two-pass, high quality) |
| `bnf` | `-o 11:bnf` | CSR-native BNF start node + deterministic parallel CM BFS, faster reorder |

The BNF variant uses George-Liu pseudo-peripheral node finder with RCM++ width-minimizing criterion and a speculative parallel Cuthill-McKee BFS that produces the same ordering as serial.

### GraphBrewOrder Ordering Strategies (Algorithm 12)

GraphBrewOrder uses Leiden community detection, then applies per-community reordering.
Options can be passed directly — the `graphbrew` prefix is **not required**.

**Ordering strategies** (passed directly as `-o 12:strategy`):

| Strategy | Example | Description |
|----------|---------|-------------|
| (default) | `-o 12` | Leiden + per-community RabbitOrder (LAYER mode) |
| `hrab` | `-o 12:hrab` | Hybrid Leiden+RabbitOrder **(best locality)** |
| `dfs` | `-o 12:dfs` | DFS dendrogram traversal |
| `bfs` | `-o 12:bfs` | BFS dendrogram traversal |
| `conn` | `-o 12:conn` | Connectivity BFS within communities |
| `dbg` | `-o 12:dbg` | DBG within each community |
| `corder` | `-o 12:corder` | Hot/cold within communities |
| `dbg-global` | `-o 12:dbg-global` | DBG across all vertices |
| `corder-global` | `-o 12:corder-global` | Hot/cold across all vertices |
| `streaming` | `-o 12:streaming` | Leiden + lazy aggregation |
| `lazyupdate` | `-o 12:lazyupdate` | Batched community weight updates |
| `rabbit` | `-o 12:rabbit` | RabbitOrder single-pass pipeline |
| `rabbit:dfs` | `-o 12:rabbit:dfs` | RabbitOrder + DFS post-ordering |

**Resolution modes:**

| Mode | Syntax | Description |
|------|--------|-------------|
| Auto | `-o 12` or `-o 12:hrab` | Compute resolution from graph density/CV (default) |
| Dynamic | `-o 12:dynamic` | Auto initial, adjust each pass |
| Fixed | `-o 12:0.75` or `-o 12:hrab:0.75` | Use specified resolution value |

**Cluster variants** (for per-community dispatch mode):

| Variant | Description | Default Final Algo |
|---------|-------------|--------------------|
| `leiden` | Leiden community detection (default) | RabbitOrder (8) |
| `rabbit` | Full RabbitOrder via GraphBrew pipeline | N/A (single-pass) |
| `hubcluster` | Hub-degree based clustering | N/A (native) |

Override the final reordering algorithm with `:<algo_id>`, e.g. `-o "12:leiden:7"` uses HubClusterDBG.

**Multi-Layer Configuration:** GraphBrewOrder's CLI string is parsed as a multi-layer pipeline:
- **Layer 0** (Preset): `leiden` | `rabbit` | `hubcluster`
- **Layer 1** (Ordering): `hrab` | `dfs` | `bfs` | `conn` | `dbg` | `corder` | `dbg-global` | `corder-global` | `streaming` | `lazyupdate` | ...
- **Layer 2** (Aggregation): `gvecsr` | `leiden` | `streaming` | `hybrid`
- **Layer 3** (Features): `merge` | `hubx` | `gord` | `hsort` | `rcm` | `norefine` | `verify` | `graphbrew` | `recursive` | `flat` (additive — combine any)
- **Layer 4** (Dispatch): `finalAlgoId` (0-11), `depth` (-1=auto), `subAlgoId`
- **Layer 5** (Numeric): resolution (float), max_iterations, max_passes

Example: `-o 12:leiden:hrab:gvecsr:merge:hubx:0.75` sets preset=leiden, ordering=hrab, aggregation=gvecsr, features=[merge,hubx], resolution=0.75.

See `GRAPHBREW_LAYERS` in `scripts/lib/core/utils.py` for the full definition.

**Chained reorderings:** Pass multiple `-o` flags to apply orderings sequentially:
```bash
./bench/bin/converter -f graph.el -s -o 2 -o 8:csr -b graph.sg  # SORT then RABBITORDER
```
See [[Reordering-Algorithms#chained-orderings-multi-pass]] for all defined chains.

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
./bench/bin/pr -f graph.el -s -o 12 -n 5
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
| `-d <delta>` | Delta for delta-stepping | 1 |

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
| `-V <file>` | Output separate CSR array files (.out_degree/.out_neigh/.offset) |

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
# Override default weights file
export PERCEPTRON_WEIGHTS_FILE=/path/to/weights.json
./bench/bin/pr -f graph.el -s -o 14 -n 3
```

**Note:** If not set, AdaptiveOrder loads weights from `adaptive_models.json` via the DB hook (see [[Perceptron-Weights#weight-file-location]]).

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
Graph has N nodes and M edges
Reordering with HUBCLUSTERDBG...

Trial   Time(s)
1       X.XXXX
2       X.XXXX
3       X.XXXX
4       X.XXXX
5       X.XXXX

Average: X.XXXX seconds
Std Dev: X.XXXX seconds
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
1       X.XXXX    ...             ...
2       X.XXXX    ...             ...
3       X.XXXX    ...             ...

Average: X.XXXX seconds, XX.X MTEPS
```

### AdaptiveOrder Output

```
=== Adaptive Reordering Selection ===
Comm    Nodes   Edges   Density Selected
...     ...     ...     ...     ...

=== Algorithm Selection Summary ===
(shows how many communities selected each algorithm)
```

---

## Common Command Patterns

### Quick Test

```bash
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -n 1
```

### Compare Algorithms

```bash
for algo in 0 7 12 14 15; do
    echo "=== Algorithm $algo ==="
    ./bench/bin/pr -f graph.el -s -o $algo -n 3
done
```

### Run All Benchmarks

```bash
for bench in pr pr_spmv bfs cc cc_sv sssp bc tc; do
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
| Variants | `--all-variants`, `--rabbit-variants LIST`, `--gorder-variants LIST`, `--graphbrew-variants LIST`, `--resolution VALUE` |
| Weights | *(automatic via `adaptive_models.json`)* |
| Labels | `--precompute`, `--generate-maps`, `--use-maps` |
| Validation | `--brute-force`, `--validation-benchmark NAME` |
| Dependencies | `--check-deps`, `--install-deps`, `--install-boost` |

```bash
# Quick examples
python3 scripts/graphbrew_experiment.py --full --size small --auto --quick
python3 scripts/graphbrew_experiment.py --train --size medium --auto --precompute
python3 scripts/graphbrew_experiment.py --brute-force --validation-benchmark bfs
```

### eval_weights

Trains weights and simulates C++ scoring to report accuracy/regret. See [[Python-Scripts#-weight-evaluation---eval-weights]].

```bash
python3 scripts/graphbrew_experiment.py --eval-weights  # No arguments needed
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
