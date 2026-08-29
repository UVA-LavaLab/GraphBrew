# Running Benchmarks

Complete guide to running GraphBrew benchmarks with all options explained.

## Overview

The automated pipeline runs **seven** benchmarks by default (`EXPERIMENT_BENCHMARKS`). Triangle Counting (TC) is excluded because it is a combinatorial counting kernel that does not benefit from vertex reordering the way traversal-style algorithms do. Use `--benchmarks pr bfs tc` to opt-in to TC or any specific subset.

| Benchmark | Binary | Description |
|-----------|--------|-------------|
| PageRank (pull) | `pr` | Page importance ranking |
| PageRank (SpMV) | `pr_spmv` | Sparse matrix-vector PageRank |
| BFS | `bfs` | Breadth-First Search traversal |
| Connected Components (Afforest) | `cc` | Find graph connectivity |
| Connected Components (SV) | `cc_sv` | Shiloach-Vishkin CC |
| SSSP | `sssp` | Single-Source Shortest Paths |
| Betweenness Centrality | `bc` | Node importance by path flow |

> **Available but excluded by default:** Triangle Counting (`tc`) — add via `--benchmarks pr bfs cc sssp tc`.

> **Shuffled control:** By default, `.mtx` graphs are converted to `.sg` with
> the fixed seeded RANDOM ordering. This is a controlled labeling, not a
> worst-case claim. Disable with `--no-random-baseline`.

> **Pre-generated Reordered .sg:** After RANDOM baseline conversion, each algorithm's reordered graph is pre-generated as `{graph}_{ALGO}.sg` (e.g., `email-Enron_SORT.sg`, `email-Enron_HUBCLUSTERDBG.sg`). At benchmark time, the pre-generated `.sg` is loaded with `-o 0` (ORIGINAL), eliminating runtime reorder overhead. Disk space is estimated first; if insufficient, the pipeline falls back to real-time reordering. Control with `--pregenerate-sg` (default ON) / `--no-pregenerate-sg`.

---

## Automated Pipeline (Recommended)

For batch benchmarking, use the unified experiment script:

Pass `--graphs-dir /media/NVMeData/00_GraphDatasets/GraphBrew` on the evaluation
host so downloaded or converted graphs do not fill the repository filesystem.

```bash
# One-command: download graphs and run the generic collection pipeline
python3 scripts/graphbrew_experiment.py --target-graphs 150

# Preview what would run (no execution)
python3 scripts/graphbrew_experiment.py --target-graphs 150 --dry-run

# Full generic pipeline with explicit flags
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

```

See [[Command-Line-Reference]] for phases, download size options, and
memory/disk management. See [[Python-Scripts]] for module documentation and
[[Reproducible-Experiments]] for the isolated frozen-study matrix.

### Frozen Study Reproduction

Use the top-level orchestrator for normal preview and full runs:

```bash
# Preview
python3 scripts/graphbrew_experiment.py --vldb --paper-preview \
  --paper-graph-dir /media/NVMeData/00_GraphDatasets/GraphBrew \
  --paper-artifact-root /media/NVMeData/00_GraphDatasets/GraphBrew/artifacts \
  --paper-threads 4 --paper-cpu-list 24-27

# Full frozen matrix
python3 scripts/graphbrew_experiment.py --vldb \
  --paper-graph-dir /media/NVMeData/00_GraphDatasets/GraphBrew \
  --paper-artifact-root /media/NVMeData/00_GraphDatasets/GraphBrew/artifacts \
  --paper-threads 16 --paper-cpu-list 0-15
```

Use `scripts/experiments/vldb/stages/` only for restartable long runs.

### Full Collection Example

Here is exactly what happens when you run:

```bash
python3 scripts/graphbrew_experiment.py --target-graphs 150 --size small
```

**What `--target-graphs 150 --size small` does:**
- Auto-enables `--full`, `--catalog-size 150`, `--auto`, `--all-variants`
- Limits to the `small` size category (10K–500K edges)

**Pipeline phases (in order):**

| Phase | What Happens | Output |
|-------|-------------|--------|
| **Download** | Fetches up to 150 small graphs from SuiteSparse (16 hardcoded + auto-discovered) | `results/graphs/<name>/<name>.mtx` |
| **Build** | Compiles C++ benchmark binaries (standard + cache sim) | `bench/bin/pr`, `bench/bin/bfs`, ... |
| **Convert** | Converts `.mtx` → `.sg` with RANDOM baseline ordering | `results/graphs/<name>/<name>.sg` |
| **Pre-generate** | Creates reordered `.sg` per algorithm (MAP mode — no runtime overhead) | `results/graphs/<name>/<name>_<algo>.sg` |
| **Reorder** | Runs 17 algorithms × 14 variants on each graph → `.lo` label maps | `results/mappings/<name>/<algo>.lo` |
| **Benchmark** | Runs 7 kernels (PR, PR_SPMV, BFS, CC, CC_SV, SSSP, BC) × all orderings × 2 trials | `results/data/benchmarks.json` |
| **Cache Sim** | Simulates L1/L2/L3 cache hit rates for PR and BFS | `results/data/benchmarks.json` (cache fields) |
| **Compatibility export** | Optionally writes offline model artifacts; not a paper claim | `results/data/adaptive_models.json` |

Timelines are workload- and machine-dependent; use `--dry-run` and the
orchestrator's budget/planning modes before broad collection.

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
| `-o <id>` | Ordering algorithm (0-16) | 0 (none) |
| `-s` | Symmetrize graph (make undirected) | Off |
| `-g <scale>` | Generate 2^scale kronecker graph | - |
| `-n <num>` | Number of trials | 16 |
| `--pregenerate-sg` | Pre-generate reordered `.sg` per algorithm (eliminates runtime reorder overhead) | ON |
| `--no-pregenerate-sg` | Disable pre-generation; reorder at benchmark time instead | - |

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

## Benchmark Examples

```bash
# PageRank (iterative convergence)
./bench/bin/pr -f graph.el -s -o 7 -n 5

# BFS from vertex 0 (traversal)
./bench/bin/bfs -f graph.el -s -r 0 -o 12 -n 5

# Connected Components
./bench/bin/cc -f graph.el -s -n 5

# SSSP (requires weighted edges, .wel format)
./bench/bin/sssp -f graph.wel -s -r 0 -d 2 -n 5

# Betweenness Centrality
./bench/bin/bc -f graph.el -s -r 0 -n 5

# Triangle Counting
./bench/bin/tc -f graph.el -s -o 7 -n 5
```

See [[Command-Line-Reference]] for complete option reference, output format details, and batch scripting patterns.

---

## Reordering Options

See [[Command-Line-Reference#reordering-algorithm-ids]] for the full algorithm table (IDs 0-16) and variant syntax.

| Purpose | Algorithm | ID |
|----------|-----------|-----|
| no-reorder baseline | ORIGINAL | 0 |
| cheap degree/bucket control | HUBCLUSTERDBG | 7 |
| exact hand-configured composition | GraphBrewOrder | 12 |
| historical reuse-1/2 portfolio reproduction | AdaptiveOrder | 14 |
| bandwidth-oriented control | RCM-BNF | `11:bnf` |

Do not choose from graph-domain labels such as “social” or “road.” Start with
[Evidence and Claims](Evidence-and-Claims) and the
[GraphBrew Running Example](GraphBrew-Running-Example), then select an explicit
mechanism.

---

## See Also

- [[Command-Line-Reference]] — Full CLI options, batch scripts, output formats, environment variables
- [[Python-Scripts]] — Orchestration scripts for automated experiments

---

## Amortization Analysis

After running benchmarks, analyze whether reordering pays off:

```bash
# Amortization report (runs automatically after benchmark phase)
python3 scripts/graphbrew_experiment.py --phase all

# Standalone amortization analysis
python3 -m scripts.lib.analysis.metrics --results-dir results/

# Compare two algorithms head-to-head
python3 -m scripts.lib.analysis.metrics --results-dir results/ \
  --compare RABBITORDER_csr GraphBrewOrder_graphbrew:hrab
```

The report shows for each (graph, algorithm, benchmark):
- **Amortization iterations** — kernel runs needed to recoup reorder cost
- **E2E speedup** — speedup including reorder cost at 1, 10, 100 iterations
- **Verdict** — INSTANT (<1), FAST (1–10), OK (10–100), SLOW (>100), NEVER

See [[Python-Scripts#libanalysismetricspy---amortization--end-to-end-evaluation]] for full documentation.

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

### Pre-generated .sg: Disk Space vs Speed

Pre-generating reordered `.sg` files trades disk space for benchmark speed. Each `.sg` is roughly the same size as the RANDOM baseline, so 13 algorithms × N graphs can require significant storage. The pipeline estimates required disk space before pre-generating (`estimate_pregeneration_size()`) and falls back to real-time reordering if space is insufficient. Use `--no-pregenerate-sg` to skip pre-generation on disk-constrained systems.

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
- [AdaptiveOrder](AdaptiveOrder) - historical selector compatibility
- [[Supported-Graph-Formats]] - Input format details

---

[← Back to Home](Home) | [Graph Benchmarks →](Graph-Benchmarks)
