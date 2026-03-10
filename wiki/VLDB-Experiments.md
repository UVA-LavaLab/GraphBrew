# VLDB 2026 Experiment Guide

Complete reference for reproducing the GraphBrew **multilayered reordering** paper experiments.
This document explains how to run all experiments, generate figures, and reproduce
every number in the paper from an **empty `results/` folder**.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Prerequisites](#2-prerequisites)
3. [Experiment Overview](#3-experiment-overview)
4. [Running Experiments](#4-running-experiments)
5. [Generated Outputs](#5-generated-outputs)
6. [Configuration Reference](#6-configuration-reference)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Quick Start

```bash
# One command reproduces all paper figures:
python3 scripts/experiments/vldb_paper_experiments.py \
    --all --graph-dir /path/to/graphs

# Preview mode (fast validation, small graphs):
python3 scripts/experiments/vldb_paper_experiments.py \
    --all --preview --graph-dir /path/to/graphs

# Dry run (show commands without executing):
python3 scripts/experiments/vldb_paper_experiments.py \
    --all --dry-run

# Regenerate figures from existing results:
python3 scripts/experiments/vldb_paper_experiments.py --figures-only
```

---

## 2. Prerequisites

### Build

```bash
# On Linux (native):
make all RABBIT_ENABLE=1
make sim   # cache simulation binaries

# On Windows (via WSL):
.\build_wsl.ps1 all
```

### Graph Data

Download the evaluation graphs and convert to `.sg` format:

```bash
python3 scripts/graphbrew_experiment.py --full --size medium
```

Or manually place `.sg` files in a directory and pass `--graph-dir <path>`.

### Python Dependencies

```bash
pip install matplotlib numpy  # optional: for figure generation
```

---

## 3. Experiment Overview

The paper's evaluation consists of 6 subsections, each mapped to specific
experiments in the runner:

| § | Paper Subsection | Experiment | What It Measures |
|---|-----------------|------------|------------------|
| 4.2 | Cache Performance | Exp 1 | Cache miss rates across cache sizes (PR, all reorderings) |
| 4.3 | Kernel Speedup | Exp 2 | Algorithm execution time normalized to Original (7 benchmarks) |
| 4.4 | Overhead & E2E | Exp 3+4 | Reorder preprocessing time + amortization analysis |
| 4.5 | Sensitivity & Composability | Exp 5+6+7 | Graph-type sensitivity, layer ablation, chained orderings |
| 4.6 | Scalability | Exp 8 | Thread scaling of reorder step (1–32 threads) |

### Algorithms Evaluated

**Baselines (11):** Original, Random, SORT, HubSort, HubCluster, DBG,
HubSortDBG, HubClusterDBG, RabbitOrder, Gorder, RCM, GoGraph

**GraphBrew Variants (7):** Leiden, Rabbit, HubCluster, HRAB, TQR, HCache, Streaming

**Chained Orderings (5):** SORT→RabbitOrder, SORT→GB-Leiden,
DBG→GB-Leiden, SORT→GB-HRAB, HubClusterDBG→RabbitOrder

### Benchmark Algorithms (7)

BFS, PR (PageRank), PR-SpMV, SSSP, CC (Afforest), CC-SV, BC

### Evaluation Graphs (9)

| Graph | Vertices (M) | Edges (M) | Type |
|-------|------------:|----------:|------|
| cit-Patents | 6.01 | 16.52 | Citation |
| soc-pokec | 1.63 | 30.62 | Social |
| USA-Road | 23.95 | 58.33 | Road |
| soc-LiveJournal1 | 4.85 | 68.99 | Social |
| com-orkut | 3.07 | 117.19 | Social |
| wikipedia_link_en | 12.15 | 378.14 | Content |
| Gong-gplus | 28.94 | 462.99 | Social |
| webbase-2001 | 118.14 | 1,019.90 | Web |
| twitter | 61.79 | 1,468.36 | Social |

---

## 4. Running Experiments

### Full Evaluation

```bash
# Run all 8 experiments + auto-generate figures:
python3 scripts/experiments/vldb_paper_experiments.py \
    --all --graph-dir /data/graphs

# Run specific experiments (e.g., cache + speedup only):
python3 scripts/experiments/vldb_paper_experiments.py \
    --exp 1 2 --graph-dir /data/graphs

# Skip figure generation:
python3 scripts/experiments/vldb_paper_experiments.py \
    --all --graph-dir /data/graphs --no-figures
```

### Preview Mode

For fast validation before the full run:

```bash
python3 scripts/experiments/vldb_paper_experiments.py \
    --all --preview --graph-dir /data/graphs
```

Preview uses: 2 small graphs, 1 trial, 2 benchmarks (PR, BFS), 300s timeout.

### Custom Graph Set

```bash
python3 scripts/experiments/vldb_paper_experiments.py \
    --all --graphs cit-Patents soc-pokec --graph-dir /data/graphs
```

### Figure Generation Only

```bash
# From real experiment data:
python3 scripts/experiments/vldb_paper_experiments.py --figures-only

# With sample/placeholder data (for layout preview):
python3 scripts/experiments/vldb_generate_figures.py --sample-data
```

---

## 5. Generated Outputs

```
results/vldb_paper/
├── MANIFEST.json              # Reproducibility metadata (git hash, config, timing)
├── exp1_cache/                # Cache simulation results (JSON)
├── exp2_speedup/              # Kernel speedup results (JSON)
├── exp3_overhead/             # Reorder overhead results (JSON)
├── exp4_e2e/                  # End-to-end derived data
├── exp5_ablation/             # Ablation study results (JSON)
├── exp6_sensitivity/          # Graph-type sensitivity metadata
├── exp7_chained/              # Chained ordering results (JSON)
├── exp8_scalability/          # Thread scaling results (JSON)
├── figures/                   # Generated PNG figures
│   ├── fig1_cache_performance.png
│   ├── fig2_kernel_speedup.png
│   └── fig3_reorder_overhead.png
└── tables/                    # Generated LaTeX table snippets
    ├── table_variants.tex
    ├── table_ablation.tex
    ├── table_sensitivity.tex
    └── table_chained.tex
```

Figures are also copied to the paper's `dataCharts/` directory for direct
`\includegraphics` inclusion.

---

## 6. Configuration Reference

All experiment parameters are defined in
`scripts/experiments/vldb_config.py`:

| Parameter | Full | Preview |
|-----------|------|---------|
| Trials | 3 | 1 |
| Benchmarks | 7 (bfs, pr, pr_spmv, sssp, cc, cc_sv, bc) | 2 (pr, bfs) |
| Graphs | 9 | 2 |
| Timeout (per command) | 3600s | 300s |
| Thread counts (scaling) | 1, 2, 4, 8, 16, 32 | 1, 2, 4, 8, 16, 32 |

### CLI Flags

| Flag | Description |
|------|-------------|
| `--all` | Run all 8 experiments |
| `--exp N [N ...]` | Run specific experiment(s) by number (1-8) |
| `--preview` | Use small graphs, 1 trial, 2 benchmarks |
| `--dry-run` | Print commands without executing |
| `--graph-dir PATH` | Directory containing `.sg` and `.el` graph files |
| `--graphs NAME [...]` | Override graph list by name |
| `--no-figures` | Skip automatic figure generation |
| `--figures-only` | Generate figures from existing results (no experiments) |

---

## 7. Troubleshooting

### Common Issues

**"Binary not found"** — Run `make all RABBIT_ENABLE=1` first.

**"Graph file not found"** — Ensure `--graph-dir` points to a directory with
`.sg` files matching the graph names in the config (e.g., `cit-Patents.sg`).

**"matplotlib not available"** — Install with `pip install matplotlib numpy`.
Tables will still be generated without matplotlib.

**"Timeout"** — Large graphs (twitter, webbase) may need longer timeouts.
Edit `TIMEOUT_FULL` in `vldb_config.py`.

### Extending

To add a new graph or algorithm, edit `scripts/experiments/vldb_config.py`:
- `EVAL_GRAPHS` — add graph metadata
- `BASELINE_ALGORITHMS` — add algorithm ID and name
- `GRAPHBREW_VARIANTS` — add variant string
- `CHAINED_ORDERINGS` — add (name, flags) tuple

---

*See also: [[GraphBrewOrder]], [[Running-Benchmarks]], [[Command-Line-Reference]], [[Cache-Simulation]]*
