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

The experiment script is **self-contained**: it automatically builds binaries,
downloads graphs from SuiteSparse, and converts them to `.sg` format before
running any experiments.

```bash
# Full reproducibility (builds, downloads, runs all experiments + figures):
python3 scripts/experiments/vldb_paper_experiments.py --all

# Preview mode (fast validation — 2 small graphs, 1 trial):
python3 scripts/experiments/vldb_paper_experiments.py --all --preview

# Dry run (validate commands without executing):
python3 scripts/experiments/vldb_paper_experiments.py --all --dry-run

# Regenerate figures from existing results:
python3 scripts/experiments/vldb_paper_experiments.py --figures-only
```

The auto-setup phase will:
1. **Build** standard and cache-simulation binaries via `make`
2. **Download** 9 of 11 evaluation graphs from SuiteSparse (2 require manual download — see below)
3. **Convert** downloaded `.mtx` files to `.sg` format

To skip auto-setup (if binaries and graphs are already in place):

```bash
# Uses results/graphs/ by default:
python3 scripts/experiments/vldb_paper_experiments.py --all --skip-setup

# Or specify a custom graph directory:
python3 scripts/experiments/vldb_paper_experiments.py --all --skip-setup \
    --graph-dir /path/to/graphs
```

---

## 2. Prerequisites

### System Requirements

- Linux x86-64 with GCC ≥ 7 (tested on Ubuntu 22.04)
- ≥ 16 GB RAM for full evaluation (64 GB+ recommended for webbase-2001, twitter7)
- Python ≥ 3.8

### Automatic Steps (handled by `--all`)

The script calls `make -j$(nproc)` and `make all-sim -j$(nproc)` automatically.
If you prefer to build manually:

```bash
make all RABBIT_ENABLE=1      # standard benchmark binaries
make all-sim                   # cache simulation binaries
pip install matplotlib numpy   # optional: figure generation
```

### Manual-Download Graphs (2 of 11)

Nine evaluation graphs are downloaded automatically from SuiteSparse. Two
require manual preparation:

#### wikipedia\_link\_en

Source: [KONECT — Wikipedia link (en)](http://konect.cc/networks/wikipedia_link_en/)

Download the dataset, extract it, and convert the edge list to a file the
converter can read (tab-separated edge list → `.el`):

```bash
mkdir -p results/graphs/wikipedia_link_en
# Download from KONECT, extract, and rename to .el
# Place the edge-list file at:
#   results/graphs/wikipedia_link_en/wikipedia_link_en.el
```

#### Gong-gplus

Source: [Duke University — Google+ Social Networks](https://people.duke.edu/~zg70/gplus.html)
([Google Drive link](https://drive.google.com/file/d/1HF8Q2N_hxsaQ26MarKYxZEQhqI66qAxV/view))

The dataset contains 4 temporal snapshots. To reconstruct snapshot 4
(28.9M vertices, 463M edges), keep all edges with TimeID 0–3:

```bash
mkdir -p results/graphs/Gong-gplus
# 1. Download from the Google Drive link above
# 2. Extract and keep all directed social links (TimeID 0–3)
# 3. Strip the TimeID column to produce a two-column edge list
# 4. Place as: results/graphs/Gong-gplus/Gong-gplus.el
```

> **Note:** The auto-setup will print clear instructions for any missing
> manual-download graphs and proceed with the available ones.

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

**Baselines (12):** Original, Random, SORT, HubSort, HubCluster, DBG,
HubSortDBG, HubClusterDBG, RabbitOrder, Gorder, RCM, GoGraph

**GraphBrew Variants (7):** Leiden, Rabbit, HubCluster, HRAB, TQR, HCache, Streaming

**Chained Orderings (5):** SORT→RabbitOrder, SORT→GB-Leiden,
DBG→GB-Leiden, SORT→GB-HRAB, HubClusterDBG→RabbitOrder

### Benchmark Algorithms (7)

BFS, PR (PageRank), PR-SpMV, SSSP, CC (Afforest), CC-SV, BC

### Evaluation Graphs (11)

| Graph | Vertices (M) | Edges (M) | Type |
|-------|------------:|----------:|------|
| cit-Patents | 6.01 | 16.52 | Citation |
| soc-pokec | 1.63 | 30.62 | Social |
| USA-road-d.USA | 23.95 | 58.33 | Road |
| soc-LiveJournal1 | 4.85 | 68.99 | Social |
| delaunay_n24 | 16.78 | 100.66 | Mesh |
| hollywood-2009 | 1.14 | 113.89 | Collaboration |
| com-Orkut | 3.07 | 117.19 | Social |
| wikipedia_link_en | 12.15 | 378.14 | Content |
| Gong-gplus | 28.94 | 462.99 | Social |
| webbase-2001 | 118.14 | 1,019.90 | Web |
| twitter7 | 61.79 | 1,468.36 | Social |

---

## 4. Running Experiments

### Full Evaluation

```bash
# Run all 8 experiments (auto-setup included):
python3 scripts/experiments/vldb_paper_experiments.py --all

# Run all experiments with graphs in a specific directory:
python3 scripts/experiments/vldb_paper_experiments.py \
    --all --skip-setup --graph-dir /data/graphs

# Run specific experiments (e.g., cache + speedup only):
python3 scripts/experiments/vldb_paper_experiments.py \
    --exp 1 2

# Skip figure generation:
python3 scripts/experiments/vldb_paper_experiments.py \
    --all --no-figures
```

### Preview Mode

For fast validation before the full run:

```bash
python3 scripts/experiments/vldb_paper_experiments.py --all --preview
```

Preview uses: 2 small graphs, 1 trial, 2 benchmarks (PR, BFS), 300s timeout.

### Custom Graph Set

```bash
python3 scripts/experiments/vldb_paper_experiments.py \
    --all --graphs cit-Patents soc-pokec
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
│                              #   Per-record fields: timing (average_time, reorder_time, …)
│                              #   + L1/L2/L3 cache metrics (l1_hits, l1_misses, l1_hit_rate, …)
├── exp2_speedup/              # Kernel speedup results (JSON)
├── exp3_overhead/             # Reorder overhead results (JSON, .sg input with .el fallback)
├── exp4_e2e/                  # End-to-end derived data
├── exp5_ablation/             # Ablation study results (JSON)
├── exp6_sensitivity/          # Graph-type sensitivity metadata
├── exp7_chained/              # Chained ordering results (JSON)
├── exp8_scalability/          # Thread scaling results (JSON, .sg input with .el fallback)
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
| Graphs | 11 | 2 |
| Timeout (per command) | 3600s | 300s |
| Thread counts (scaling) | 1, 2, 4, 8, 16, 32 | 1, 2, 4, 8, 16, 32 |

### CLI Flags

| Flag | Description |
|------|-------------|
| `--all` | Run all 8 experiments |
| `--exp N [N ...]` | Run specific experiment(s) by number (1-8) |
| `--preview` | Use small graphs, 1 trial, 2 benchmarks |
| `--dry-run` | Print commands without executing |
| `--graph-dir PATH` | Directory containing graph files (default: `results/graphs` with `--skip-setup`) |
| `--graphs NAME [...]` | Override graph list by name |
| `--skip-setup` | Skip the auto-setup phase (build, download, convert) |
| `--skip-download` | Skip graph download but still build + convert |
| `--no-figures` | Skip automatic figure generation |
| `--figures-only` | Generate figures from existing results (no experiments) |

---

## 7. Troubleshooting

### Common Issues

**"Binary not found"** — The script builds binaries automatically.
If auto-build fails, run `make all RABBIT_ENABLE=1 && make all-sim` manually.

**"Graph file not found"** — Either let auto-setup download the graphs, or
ensure `--graph-dir` points to a directory with `.sg` files matching the graph
names in the config. Both flat layout (`cit-Patents.sg`) and nested layout
(`cit-Patents/cit-Patents.sg`) are supported. Experiments 3 and 8 try `.sg`
first and fall back to `.el` automatically.

**Graphs that need manual download** — `wikipedia_link_en` (KONECT) and
`Gong-gplus` (Google Drive) cannot be auto-downloaded. See
[Prerequisites §2](#2-prerequisites) for download instructions. The script will
skip these graphs and proceed with the rest.

**"matplotlib not available"** — Install with `pip install matplotlib numpy`.
Tables will still be generated without matplotlib.

**"Timeout"** — Large graphs (twitter7, webbase-2001) may need longer timeouts.
Edit `TIMEOUT_FULL` in `vldb_config.py`.

### Extending

To add a new graph or algorithm, edit `scripts/experiments/vldb_config.py`:
- `EVAL_GRAPHS` — add graph metadata
- `BASELINE_ALGORITHMS` — add algorithm ID and name
- `GRAPHBREW_VARIANTS` — add variant string
- `CHAINED_ORDERINGS` — add (name, flags) tuple

---

### Result JSON Schema

All experiment JSON files share a common set of timing fields extracted by
`parse_timing()`: `trial_time`, `reorder_time`, `average_time`,
`preprocessing_time`, `total_time`, `topology_analysis_time`, `read_time`,
`relabel_map_time`.

Experiment 1 additionally includes per-cache-level metrics extracted by
`parse_cache_sim()`: `l1_hits`, `l1_misses`, `l1_hit_rate`, `l2_hits`,
`l2_misses`, `l2_hit_rate`, `l3_hits`, `l3_misses`, `l3_hit_rate`,
`total_accesses`, `memory_accesses`, `overall_hit_rate`.

LaTeX tables (`table_ablation.tex`, `table_sensitivity.tex`,
`table_chained.tex`) are populated from the JSON data automatically;
fields that have no data yet show `\emph{TBD}`.

---

*See also: [[GraphBrewOrder]], [[Running-Benchmarks]], [[Command-Line-Reference]], [[Cache-Simulation]], [[Python-Scripts]]*
