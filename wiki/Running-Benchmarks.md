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
| `-o <id>` | Ordering algorithm (0-16) | 0 (none) |
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

# Triangle Counting (benefits most from reordering)
./bench/bin/tc -f graph.el -s -o 7 -n 5
```

See [[Command-Line-Reference]] for complete option reference, output format details, and batch scripting patterns.

---

## Reordering Options

See [[Command-Line-Reference#reordering-algorithm-ids]] for the full algorithm table (IDs 0-16) and variant syntax.

| Use Case | Algorithm | ID |
|----------|-----------|-----|
| General purpose | HUBCLUSTERDBG | 7 |
| Social networks | LeidenOrder | 15 |
| Unknown graphs | AdaptiveOrder | 14 |
| Maximum locality | LeidenCSR | 16 |
| Road networks | RCM | 11 |

---

## See Also

- [[Command-Line-Reference]] — Full CLI options, batch scripts, output formats, environment variables
- [[Python-Scripts]] — Orchestration scripts for automated experiments

---

## Amortization Analysis

After running benchmarks, analyze whether reordering pays off:

```bash
# Amortization report from latest results
python3 scripts/analyze_metrics.py --results-dir results/

# Compare two algorithms head-to-head
python3 scripts/analyze_metrics.py --results-dir results/ \
  --compare RABBITORDER_csr LeidenCSR_graphbrew:hrab
```

The report shows for each (graph, algorithm, benchmark):
- **Amortization iterations** — kernel runs needed to recoup reorder cost
- **E2E speedup** — speedup including reorder cost at 1, 10, 100 iterations
- **Verdict** — INSTANT (<1), FAST (1–10), OK (10–100), SLOW (>100), NEVER

See [[Python-Scripts#analyze_metrics.py]] for full documentation.

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
