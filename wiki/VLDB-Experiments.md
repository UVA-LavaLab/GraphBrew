# VLDB 2026 Experiment Guide

Complete reference for reproducing the GraphBrew **AdaptiveOrder-ML** experiments.
This document explains every number, every ML model, every feature, and every
pipeline phase so you can run the full experiment from an **empty `results/`
folder** and understand every output file produced.

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Numbers at a Glance](#2-numbers-at-a-glance)
3. [The 7-Phase Training Pipeline](#3-the-7-phase-training-pipeline)
4. [The 8 VLDB Experiments](#4-the-8-vldb-experiments)
5. [ML Models Explained](#5-ml-models-explained)
6. [The 22-Feature Vector](#6-the-22-feature-vector)
7. [Running from Scratch](#7-running-from-scratch)
8. [Interpreting Results](#8-interpreting-results)
9. [Variant System](#9-variant-system)

---

## 1. Background & Motivation

### The Problem

Graph algorithms (PageRank, BFS, SSSP, CC, …) access vertices in irregular
patterns that defeat hardware prefetchers and caches.  **Vertex reordering**
relabels the graph so that frequently co-accessed vertices sit close in memory,
dramatically improving cache hit rates and execution time.

The catch: **no single reordering algorithm wins on every graph**.

- **Gorder** excels on dense social networks but is too slow for billion-edge
  web graphs.
- **RabbitOrder** is fast but loses to simpler methods on road networks.
- **GraphBrewOrder** (Leiden + per-community reordering) adapts well to
  community-heavy graphs but is overkill for uniform meshes.

This is the **oracle gap** — the difference between always using a fixed
algorithm and always picking the best one per graph.  Our VLDB experiments
quantify this gap across hundreds of real-world graphs from the SuiteSparse
Matrix Collection and show that **AdaptiveOrder** (algorithm 14), an
ML-powered selector, closes it.

### The GraphBrew Approach

1. **Train** — Run all 13 reordering algorithms on a diverse corpus of graphs,
   benchmark 7 kernels, collect execution times + graph properties.
2. **Learn** — Train a perceptron (and auxiliary models) to predict the fastest
   algorithm for a new, unseen graph based on 22 topological features.
3. **Deploy** — At runtime, AdaptiveOrder extracts features from the input
   graph in ≈0.1 s, scores each algorithm, and picks the one with the highest
   predicted speedup — all inside the C++ binary, zero Python dependency.

### Why VLDB

We evaluate on **real-world sparse matrices** from SuiteSparse (up to 603
graph-like matrices), use a fully automated pipeline
(`graphbrew_experiment.py`) that handles downloads, builds, benchmarks, and ML
evaluation, and report reproducible numbers via Leave-One-Graph-Out (LOGO)
cross-validation.

---

## 2. Numbers at a Glance

| What | Count | Details |
|------|------:|---------|
| Algorithm IDs | 17 | IDs 0–16 |
| Benchmark-eligible | 15 | All except MAP (13) and AdaptiveOrder (14) |
| Reorder-producing | 13 | Eligible minus ORIGINAL (0) and RANDOM (1) |
| Benchmarks available | 8 | PR, PR_SPMV, BFS, CC, CC_SV, SSSP, BC, TC |
| Default benchmarks | 7 | TC excluded (combinatorial, not cache-sensitive) |
| Features (linear) | 16 | Topology, locality, convergence |
| Features (quadratic) | 5 | Cross-terms for non-linear interactions |
| Features (total) | 22 | Fed to perceptron; DT uses the same 22 features |
| ML models | 6 | Perceptron, DT, Hybrid, RF, XGBoost, Database kNN |
| Size categories | 4 | SMALL, MEDIUM, LARGE, XLARGE |
| SuiteSparse graphs | ~603 | Square, graph-like (kind ∈ {graph, network, multigraph}) |
| Auto-discovered | ~466 | Captured by current size bounds |
| LOGO CV folds | N | One fold per graph (leave-one-out) |
| Perceptron restarts | 5 | Random restarts for escaping local optima |
| Perceptron epochs | 800 | Per restart |
| Default trials | 3 | Benchmark repetitions per (graph, algorithm, kernel) |

### Size Categories

| Category | Edge Range | Approx. Available | Download Size |
|----------|-----------|------------------:|---------------|
| SMALL | 10 K – 500 K | ~225 | ~100 MB |
| MEDIUM | 500 K – 5 M | ~134 | ~1.1 GB |
| LARGE | 5 M – 50 M | ~70 | ~25 GB |
| XLARGE | 50 M – 500 M | ~37 | ~63 GB |

---

## 3. The 7-Phase Training Pipeline

Running `python3 scripts/graphbrew_experiment.py --target-graphs 150 --size small`
executes all 7 phases sequentially (**unless** you use `--phase` to select
specific ones):

```
┌──────────┐    ┌───────┐    ┌──────────┐    ┌──────────┐
│ Download │───▶│ Build │───▶│ Convert  │───▶│ Reorder  │
└──────────┘    └───────┘    └──────────┘    └──────────┘
                                                  │
                                                  ▼
                              ┌───────────┐  ┌──────────┐
                              │ Cache Sim │◀─│Benchmark │
                              └───────────┘  └──────────┘
                                   │
                                   ▼
                              ┌──────────┐
                              │ Evaluate │
                              └──────────┘
```

### Phase-by-Phase Reference

| # | Phase | What Happens | Key Output | Timeout |
|---|-------|-------------|------------|--------:|
| 0 | **Download** | Fetches `.mtx` files from SuiteSparse. 16 hardcoded SMALL graphs + auto-discovered extras via `ssgetpy`. Enforces `--max-memory` and `--max-disk` limits. | `results/graphs/<name>/<name>.mtx` | — |
| 1 | **Build** | Runs `make` in `bench/` to compile standard binaries (`bench/bin/`) and cache-simulation binaries (`bench/bin_sim/`). | `bench/bin/pr`, `bench/bin/bfs`, … | 600 s |
| 2 | **Convert** | Converts `.mtx` → `.sg` with **RANDOM** vertex ordering (algorithm 1). This is the baseline — all benchmark times measure improvement *relative to worst-case random*. | `results/graphs/<name>/<name>.sg` | — |
| 3 | **Reorder** | For each graph × each algorithm: runs the `converter` binary to produce a `.lo` (label-order mapping) file and records the reorder wall-clock time in a `.time` file. Pre-generates reordered `.sg` files if disk allows. | `results/mappings/<name>/<ALGO>.lo` | 43 200 s (12 h) |
| 4 | **Benchmark** | Runs 7 kernels (pr, pr_spmv, bfs, cc, cc_sv, sssp, bc) × all orderings × N trials. Results appended to the centralized JSON datastore. At benchmark time, pre-generated `.sg` is loaded with `-o 0` so there is no reorder overhead in the timing. | `results/data/benchmarks.json` | 600 s |
| 5 | **Cache Sim** | Runs instrumented binaries (`bench/bin_sim/`) that simulate a 3-level cache hierarchy (L1 32 KB 8-way LRU, L2, L3) for PR and BFS. Reports hit/miss rates per cache level. | Cache fields in `benchmarks.json` | 1 200 s |
| 6 | **Evaluate** | LOGO cross-validation on all ML models. For each graph *G*: trains on all graphs except *G*, predicts best algorithm for *G*, compares to oracle. Reports top-1/top-3 accuracy, mean/median/p95 regret, within-5% rate. | `results/data/evaluation_summary.json` | — |

### Why RANDOM Baseline?

If the original `.mtx` file already has a "good" ordering (e.g., BFS
numbering from the dataset author), benchmarking algorithms against it
conflates the quality of the *dataset ordering* with the quality of the
*reordering algorithm*.  Converting to RANDOM first means every algorithm
starts from the same worst-case baseline, so speedup numbers are comparable
across graphs.

### Why `.lo` Mapping Files?

Reordering algorithms can take minutes (Gorder) or hours (COrder on large
graphs).  Pre-computing the mapping once and storing it in a `.lo` file means:

- **Reproducible benchmarks** — the same mapping is reused across all trials
  and kernels.
- **Fair timing** — benchmark time measures *only* kernel execution, not
  reorder overhead (reorder cost is tracked separately in `.time` files).
- **MAP mode** — at benchmark time, the binary loads the mapping with
  `-o 13:<path>.lo` (algorithm 13 = MAP = "load from file").

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--target-graphs N` | — | Auto-sets `--full`, `--catalog-size N`, `--auto`, `--all-variants` |
| `--size` | `all` | Size category: `small`, `medium`, `large`, `xlarge`, `all` |
| `--trials` | 3 | Benchmark repetitions per (graph, algo, kernel) |
| `--benchmarks` | 7 default | Space-separated list of kernels |
| `--skip-slow` | off | Skip Gorder (9), COrder (10), RCM (11) |
| `--skip-cache` | off | Skip cache simulation phase |
| `--skip-eval` | off | Skip LOGO evaluation phase |
| `--dry-run` | off | Print pipeline plan without executing |
| `--max-memory` | auto | Maximum RAM in GB (auto = 60% of physical) |
| `--max-disk` | auto | Maximum disk in GB (auto = 80% of free) |
| `--timeout-reorder` | 43 200 | Per-algorithm reorder timeout (seconds) |
| `--timeout-benchmark` | 600 | Per-benchmark timeout (seconds) |

---

## 4. The 8 VLDB Experiments

All experiments are orchestrated by `scripts/experiments/vldb_experiments.py`.
Run them individually or all at once:

```bash
# All 8 experiments on LARGE graphs
python scripts/experiments/vldb_experiments.py --all --size large

# Specific experiments only
python scripts/experiments/vldb_experiments.py --exp 1 3 --size large

# Preview commands without executing
python scripts/experiments/vldb_experiments.py --all --dry-run

# Lightweight local preview (fewer graphs, 1 trial, 2 benchmarks)
python scripts/experiments/vldb_experiments_small.py --all
```

### Experiment 1 — Oracle Gap Analysis

**What it proves:** No single reordering algorithm dominates across all graphs.

**Method:** Run *all* 15 eligible algorithms (every ID except 13 MAP and 14
AdaptiveOrder) on every graph in the corpus.  For each graph, identify the
**oracle** (fastest algorithm).  Plot the distribution of oracle choices — if
every graph had the same winner, adaptive selection would be pointless.

```bash
python scripts/experiments/vldb_experiments.py --exp 1 --size large
```

**Expected output:** `results/vldb_experiments/exp1_oracle_gap/` — benchmark
JSON + per-graph oracle mapping.  Expect 4–6 different algorithms appearing as
oracle across the corpus.

---

### Experiment 2 — AdaptiveOrder vs Static Baselines

**What it proves:** AdaptiveOrder matches or beats every static baseline on
aggregate.

**Method:** Run AdaptiveOrder (algorithm 14) in three selection modes on the
same graph set used in Experiment 1:

| Mode | CLI Flag | Optimizes For |
|------|----------|---------------|
| 1 | `fastest_exec` | Minimize kernel execution time |
| 2 | `best_e2e` | Minimize reorder + execution |
| 3 | `best_amort` | Minimize amortized cost over repeated runs |

```bash
python scripts/experiments/vldb_experiments.py --exp 2 --size large
```

**Expected output:** `results/vldb_experiments/exp2_adaptive/` — one sub-dir
per mode.  Compare geometric mean speedup vs ORIGINAL across modes.

---

### Experiment 3 — Algorithm Selection Accuracy (LOGO CV)

**What it proves:** The ML models generalize — they correctly predict the best
algorithm for graphs not seen during training.

**Method:** Leave-One-Graph-Out cross-validation.  For *N* graphs, run *N*
folds.  In each fold, train on *N−1* graphs, evaluate on the held-out graph.
Report:

| Metric | Meaning |
|--------|---------|
| **Top-1 accuracy** | Predicted best = actual best |
| **Top-3 accuracy** | Actual best is in top 3 predictions |
| **Mean regret** | Average % slowdown vs oracle |
| **Within-5%** | Fraction of graphs where predicted is ≤ 5% slower than oracle |

```bash
python scripts/experiments/vldb_experiments.py --exp 3
# Or directly:
python scripts/evaluate_all_modes.py --all --json
```

**Expected output:** `results/vldb_experiments/exp3_logo_cv/logo_cv_results.log`

**Minimum:** ≥ 3 graphs (≥ 10 recommended for stable metrics).

---

### Experiment 4 — Feature Importance Ablation

**What it proves:** Which features (packing factor, FEF, WSR, quadratic
cross-terms) contribute most to selection accuracy.

**Method:** For each feature group, zero the corresponding weights in
AdaptiveOrder and re-run the full benchmark.  Compare accuracy to the
all-features baseline.

| Ablation | Env Variable | Target Feature |
|----------|-------------|----------------|
| `no_packing` | `ADAPTIVE_ZERO_FEATURES=packing` | Packing factor (IISWC'18) |
| `no_fef` | `ADAPTIVE_ZERO_FEATURES=fef` | Forward edge fraction (GoGraph) |
| `no_wsr` | `ADAPTIVE_ZERO_FEATURES=wsr` | Working set ratio (P-OPT) |
| `no_quadratic` | `ADAPTIVE_ZERO_FEATURES=quadratic` | All 5 cross-terms |
| `no_types` | `ADAPTIVE_NO_TYPES=1` | Graph type clustering |
| `no_ood` | `ADAPTIVE_NO_OOD=1` | Out-of-distribution guardrail |
| `no_margin` | `ADAPTIVE_NO_MARGIN=1` | Margin-based fallback |
| `no_leiden` | `ADAPTIVE_NO_LEIDEN=1` | Leiden community features |
| `all_features` | (none) | Baseline — everything enabled |

```bash
python scripts/experiments/vldb_experiments.py --exp 4 --size medium
```

**Expected output:** `results/vldb_experiments/exp4_feature_ablation/<group>/`

---

### Experiment 5 — Cold-Start → Warm Learning Curve

**What it proves:** AdaptiveOrder's streaming database reaches near-full
accuracy after observing only 10–15 graphs.

**Method:** Simulate the streaming scenario:

1. Start with an empty knowledge base.
2. Process graphs one at a time (random order).
3. After each graph, retrain the perceptron on accumulated data.
4. Evaluate selection accuracy on the remaining unseen graphs.
5. Repeat for multiple random permutations to reduce variance.

```bash
python scripts/experiments/vldb_experiments.py --exp 5 --size medium

# Then generate the learning curve:
python -m scripts.lib.analysis.cold_start_sim \
  --benchmark-db results/data/benchmarks.json \
  --output results/vldb_experiments/exp5_cold_start/learning_curve.json \
  --permutations 10
```

**Expected output:** x-axis = graphs seen, y-axis = top-1 accuracy on unseen
graphs.  Accuracy should plateau around 10–15 graphs.

---

### Experiment 6 — Cache Performance

**What it proves:** Reordering improves cache hit rates, and AdaptiveOrder
selects orderings with near-optimal cache behavior.

**Method:**

- **Part A (Simulation):** Run instrumented `bin_sim/` binaries that simulate
  L1/L2/L3 cache hierarchies for PR and BFS.
- **Part B (HW Counters):** Run with `perf` hardware counters on representative
  graphs (requires `perf stat` access).

```bash
python scripts/experiments/vldb_experiments.py --exp 6
```

**Expected output:** `results/vldb_experiments/exp6_cache/` — per-graph,
per-algorithm cache hit rates + optional perf counter data.

---

### Experiment 7 — GoGraph Convergence Analysis

**What it proves:** GoGraph's forward-edge-fraction (FEF) maximization reduces
the number of PageRank iterations needed for convergence.

**Method:** Run PageRank on representative graphs with ORIGINAL, Gorder,
GraphBrewOrder, and GoGraphOrder.  Parse iteration counts from stdout.

```bash
python scripts/experiments/vldb_experiments.py --exp 7
```

**Expected output:** `results/vldb_experiments/exp7_convergence/convergence_*.json`
— per-graph iteration counts + average times.

---

### Experiment 8 — Scalability Analysis

**What it proves:** AdaptiveOrder's overhead is near-constant (feature
extraction only) while Gorder/COrder scale super-linearly.

**Method:** Measure reorder wall-clock time across graph sizes spanning 4
orders of magnitude (10 K → 1 B+ edges).

```bash
python scripts/experiments/vldb_experiments.py --exp 8
```

**Expected output:** `results/vldb_experiments/exp8_scalability/` — reorder
time logs per graph × algorithm combination.

---

## 5. ML Models Explained

GraphBrew trains and evaluates **six** ML models.  All are trained on the same
data (`benchmarks.json` + `graph_properties.json`) and evaluated via
LOGO cross-validation.  The **perceptron** is the primary production model
because it can be fully embedded in C++ with zero runtime dependencies.

### 5.1 Perceptron (Primary)

**What it does:** One perceptron per candidate algorithm.  For a given graph,
each perceptron scores "how good is *my* algorithm for *this* graph?"  The
algorithm with the highest score wins.

**Score formula:**

$$\text{score}_a = \text{bias}_a + \sum_{i=0}^{15} w_i^{(a)} \cdot f_i + \sum_{j=0}^{4} w_{\text{cross},j}^{(a)} \cdot f_{\text{cross},j} + w_{\text{cache}} \cdot \Delta_{\text{cache}} + w_{\text{conv}} \cdot \Delta_{\text{fef}}$$

Where:
- $f_0 \ldots f_{15}$ are the 16 linear features (see Section 6)
- $f_{\text{cross},0} \ldots f_{\text{cross},4}$ are the 5 quadratic cross-terms
- $\Delta_{\text{cache}}$ = cache impact (L1/L2/L3 improvement)
- $\Delta_{\text{fef}}$ = convergence bonus (GoGraph FEF gain)
- Multiply by per-benchmark weight multiplier

**Training:**
- **Margin-based update** (Jimenez, MICRO 2016): update only when the margin
  between correct and incorrect is below threshold $\theta$
- **Averaged perceptron** (Freund & Schapire 1999): final weights = average
  over all training steps (smooths noise)
- **Adaptive theta**: $\theta = \lfloor (1.93 \times n_{\text{feat}} + 14) \times W_{\max} / 127 \rfloor \approx 5$ for 22 features
- **Hyperparameters:** 5 random restarts × 800 epochs, learning rate 0.05
  decayed by 0.997/epoch, weight cap $W_{\max} = 16.0$
- **Seed:** `42 + restart × 1000 + bench_idx × 100` (deterministic)

**Storage:** `results/models/perceptron/type_0/weights.json` (generic weights),
`results/models/perceptron/type_0/pr.json` (per-benchmark specialization)

### 5.2 Decision Tree (DT)

**What:** CART classifier (Gini impurity, max depth 6).  Uses a 12-feature
subset matching the C++ `ModelTree::extract_features()` layout so the tree can
be exported and evaluated inside C++ without Python.

**Advantage:** Fully interpretable — you can print the tree and trace each
split decision.

### 5.3 Hybrid DT + Perceptron

**What:** DT does initial algorithm selection.  Within each leaf node, a
per-leaf perceptron refines the choice using the full 22-feature vector.  This
combines the DT's coarse partitioning with the perceptron's fine-grained
scoring.

### 5.4 XGBoost

**What:** Gradient-boosted ensemble.  Family-based variants (community /
bandwidth / cache algorithm groups) allow structured multi-class classification.
Serves as a strong upper-bound on achievable accuracy.

### 5.5 Random Forest (RF)

**What:** Bagged ensemble of decision trees (100 estimators by default).
Reduces variance vs a single DT, at the cost of losing interpretability.

### 5.6 Database kNN

**What:** Oracle lookup for graphs already in the database (`benchmarks.json`).
For unknown graphs, extract features and find the *k* nearest neighbors by
normalized feature distance.  Use the best algorithm from the closest neighbor.

**Advantage:** No training — just needs a populated benchmark database.
**Disadvantage:** Accuracy depends on having similar-looking graphs in the DB.

### Model Storage Paths

| File | Contents |
|------|----------|
| `results/data/adaptive_models.json` | Unified model store (perceptron + DT + hybrid) |
| `results/models/perceptron/type_0/weights.json` | Type-0 (generic) perceptron weights |
| `results/models/perceptron/type_0/<benchmark>.json` | Per-benchmark specialized weights |
| `results/models/perceptron/registry.json` | Graph-type cluster registry (centroids) |
| `results/data/benchmarks.json` | Centralized benchmark database (append-only) |
| `results/data/graph_properties.json` | Per-graph 22-element feature vectors |
| `results/data/evaluation_summary.json` | LOGO CV results for all models |

---

## 6. The 22-Feature Vector

The perceptron ingests a **22-element feature vector** per graph, designed to
capture topology, locality, and convergence properties.  All features are
computed in the C++ binary at runtime (no Python needed for inference).

### 6.1 Linear Features (0–15)

| # | Name | Transform | What It Measures | Source |
|---|------|-----------|-----------------|--------|
| 0 | `modularity` | raw | Leiden/Louvain quality Q. High → strong community structure. CC-count × 1.5 fallback if Leiden unavailable | — |
| 1 | `degree_variance` | raw | Coefficient of variation of degree distribution. High → power-law (hubs) | — |
| 2 | `hub_concentration` | raw | Edge fraction from top 10% highest-degree vertices. High → hub-dominated | — |
| 3 | `log_nodes` | log₁₀(N+1) | Graph scale (vertices) | — |
| 4 | `log_edges` | log₁₀(E+1) | Graph scale (edges) | — |
| 5 | `density` | E/(N·(N−1)/2) | Edge density. Very small for large real-world graphs | — |
| 6 | `avg_degree` | ÷100 | Mean vertex degree, normalized | — |
| 7 | `clustering_coeff` | raw | Local clustering coefficient (sampled, 1000 nodes) | — |
| 8 | `avg_path_length` | ÷10 | Multi-source BFS (5 sources, ≤ min(100 K, N/10) nodes each) | — |
| 9 | `diameter` | ÷50 | Max BFS depth (lower bound from sampled sources) | — |
| 10 | `community_count` | log₁₀(count+1) | Connected component count (or Leiden community count) | — |
| 11 | `packing_factor` | raw | Cache-line utilization for neighbors. High → good existing locality | IISWC'18 |
| 12 | `forward_edge_fraction` | raw | Fraction of edges (u,v) where u < v. Measures natural topological sort quality | GoGraph |
| 13 | `working_set_ratio` | log₂(WSR+1) | `graph_bytes / LLC_size`. High → graph overflows last-level cache | P-OPT |
| 14 | `vertex_significance_skew` | raw | CV of per-vertex locality scores. Measures ordering inequality | DON-RL |
| 15 | `window_neighbor_overlap` | raw | Mean fraction of neighbors within a sliding window of vertex IDs | DON-RL |

### 6.2 Quadratic Cross-Terms (16–20)

These capture non-linear interactions that individual features miss:

| # | Formula | Intuition |
|---|---------|-----------|
| 16 | `degree_variance × hub_concentration` | Power-law indicator: both high → aggressive hub-sorting helps |
| 17 | `modularity × log₁₀(N)` | Scalable community structure: high modularity matters more on large graphs |
| 18 | `packing_factor × log₂(WSR+1)` | Locality-vs-capacity trade-off: good packing + LLC overflow → reordering is critical |
| 19 | `vertex_significance_skew × hub_concentration` | DON-RL cross-term: skewed locality + hubs → hub-aware methods win |
| 20 | `window_neighbor_overlap × packing_factor` | DON-RL cross-term: existing locality quality interaction |

### 6.2.5 Cache-Line Feature (21)

| # | Name | Transform | What It Measures | Source |
|---|------|-----------|-----------------|--------|
| 21 | `packing_factor_cl` | raw | Fraction of neighbors on the same 64-byte cache line as the hub vertex. Faithful IISWC'18 definition using CL_VERTS=16 | IISWC'18 |

### 6.3 Sampling Strategy

Feature extraction must be fast (< 0.5 s even on billion-edge graphs), so all
features are sampled:

| Feature Group | Sample Size |
|--------------|-------------|
| Degree stats (1, 2, 6) | max(5000, min(√N, 50000)) — strided |
| Clustering coefficient (7) | 1000 random nodes |
| BFS diameter / path length (8, 9) | 5 sources, each visiting ≤ min(100 K, N/10) nodes |
| Packing factor (11) | max(16, N/1000) cache-line windows |
| DON-RL locality (14, 15) | max(64, N/100) vertex-ID windows |
| FEF (12) | 2000 sampled vertices |

### 6.4 Graph Type Detection

A lightweight decision tree on features 0–2 and 6 classifies each graph:

| Type | Triggers |
|------|----------|
| `road` | degree_variance < 0.5, avg_degree < 5 |
| `social` | high hub_concentration (> 0.4), high modularity |
| `web` | high degree_variance, moderate modularity |
| `powerlaw` | extreme degree_variance (> 3.0) |
| `uniform` | low degree_variance, low modularity |
| `generic` | everything else |

Type-specific perceptron weights are stored in
`results/models/perceptron/type_<id>/`.  If the graph's type matches a known
cluster (distance < 0.15), type-specific weights are used; otherwise the
generic type_0 weights serve as fallback.

---

## 7. Running from Scratch

### 7.1 Prerequisites

```bash
# System dependencies
sudo apt install g++ make wget   # or brew install on macOS

# Python dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt   # ssgetpy, numpy, scikit-learn, xgboost, ...
```

### 7.2 Clean Slate

```bash
# Delete all previous results (graphs, mappings, models, benchmarks)
rm -rf results/

# Verify the script boots cleanly
python3 scripts/graphbrew_experiment.py --target-graphs 3 --size small --dry-run
```

Expected output:
```
[INFO] --target-graphs 3: enabled --full, --catalog-size 3, --auto, --all-variants, --size small
[INFO] Using --size small: graphs=small, download=SMALL
[INFO] Auto-detected memory limit: ...
============================================================
  GRAPHBREW DRY RUN — planned pipeline stages
============================================================
  1. Download      size=SMALL, catalog-size=3
  2. Build         compile C++ benchmark binaries
  3. Convert       .mtx → .sg (RANDOM baseline)
  4. Reorder       generate .lo mappings, variants=ALL
  5. Benchmark     benchmarks=pr, pr_spmv, bfs, cc, cc_sv, sssp, bc, trials=2
  6. Cache Sim     pr, bfs
  7. Evaluate      LOGO CV on all ML models
  ...
============================================================
  (Remove --dry-run to execute)
```

### 7.3 Smoke Test (~5 min)

```bash
python3 scripts/graphbrew_experiment.py \
  --full --size small --auto --skip-cache \
  --graph-list ca-GrQc email-Enron soc-Slashdot0902 \
  --benchmarks pr bfs \
  --trials 2
```

This downloads 3 tiny graphs (< 10 MB each), builds C++ binaries, runs 2
benchmarks × 2 trials, and verifies the pipeline end-to-end.

### 7.4 Small Training Run (~30 min)

```bash
python3 scripts/graphbrew_experiment.py --target-graphs 50 --size small
```

Downloads up to 50 small graphs (10 K – 500 K edges), runs all 7 phases
including cache simulation and LOGO evaluation.  Produces
`results/data/evaluation_summary.json` with ML accuracy metrics.

### 7.5 VLDB Preview (~1 hour)

```bash
python scripts/experiments/vldb_experiments_small.py --all
```

Lightweight version of all 8 experiments using 5 representative graphs, 1
trial, 2 benchmarks.  Good for verifying experiment scripts work before a
multi-hour run.

### 7.6 Full VLDB Suite (~6–12 hours)

```bash
python scripts/experiments/vldb_experiments.py --all --size large
```

All 8 experiments on the LARGE corpus (~37 graphs, ~25 GB download).  Expect
6–12 hours depending on hardware.  First run downloads graphs; subsequent runs
reuse cached data.

### 7.7 Interruption & Resume

The pipeline is **interrupt-safe**:

- **`benchmarks.json`** is append-only with per-(graph, algorithm, benchmark)
  deduplication.  If interrupted, re-running the same command skips completed
  entries.
- **`.lo` mapping files** persist in `results/mappings/`.  Already-generated
  mappings are reused.
- **`.sg` files** persist in `results/graphs/`.  Already-downloaded graphs are
  not re-downloaded.
- **Atomic saves:** JSON files are written via temp-file + rename to prevent
  corruption on crash.

Simply re-run the same command after an interruption.

### 7.8 Expected Directory Structure After a Run

```
results/
├── data/
│   ├── benchmarks.json             # All benchmark results (append-only)
│   ├── graph_properties.json       # Per-graph feature vectors
│   ├── evaluation_summary.json     # LOGO CV metrics
│   └── adaptive_models.json        # Unified model store
├── graphs/
│   ├── email-Enron/
│   │   ├── email-Enron.mtx         # Original Matrix Market file
│   │   └── email-Enron.sg          # Serialized graph (RANDOM baseline)
│   └── ...
├── mappings/
│   ├── email-Enron/
│   │   ├── SORT.lo                 # Sort mapping (vertex permutation)
│   │   ├── SORT.time               # Sort reorder wall-clock time
│   │   ├── RABBITORDER_csr.lo      # RabbitOrder CSR variant mapping
│   │   ├── GraphBrewOrder_leiden.lo
│   │   └── ...
│   └── ...
├── models/
│   └── perceptron/
│       ├── type_0/
│       │   ├── weights.json        # Generic perceptron weights
│       │   ├── pr.json             # PR-specialized weights
│       │   └── ...
│       └── registry.json           # Type cluster centroids
├── logs/
│   └── run_<timestamp>/            # Per-run logs
└── vldb_experiments/               # Per-experiment output (Exp 1–8)
    ├── exp1_oracle_gap/
    ├── exp2_adaptive/
    ├── exp3_logo_cv/
    ├── exp4_feature_ablation/
    ├── exp5_cold_start/
    ├── exp6_cache/
    ├── exp7_convergence/
    └── exp8_scalability/
```

---

## 8. Interpreting Results

### 8.1 `benchmarks.json` Schema

Each entry represents one (graph, algorithm, benchmark) measurement:

```json
{
  "graph": "email-Enron",
  "algorithm": "GraphBrewOrder_leiden",
  "algorithm_id": 12,
  "benchmark": "pr",
  "avg_time": 0.0342,
  "reorder_time": 0.085,
  "trials": [0.0351, 0.0338, 0.0337],
  "nodes": 36692,
  "edges": 367662,
  "timestamp": "2026-03-04T10:15:00"
}
```

**Deduplication:** If the same (graph, algorithm, benchmark) tuple already
exists, the entry with the faster `avg_time` wins (best-time-wins policy).

### 8.2 `graph_properties.json` Schema

Per-graph feature vectors:

```json
{
  "email-Enron": {
    "nodes": 36692,
    "edges": 367662,
    "modularity": 0.573,
    "degree_variance": 3.21,
    "hub_concentration": 0.38,
    "avg_degree": 10.02,
    "clustering_coefficient": 0.497,
    "avg_path_length": 4.03,
    "diameter": 11,
    "community_count": 234,
    "packing_factor": 0.12,
    "forward_edge_fraction": 0.51,
    "working_set_ratio": 0.8
  }
}
```

### 8.3 `evaluation_summary.json`

LOGO CV results per model × criterion:

| Field | Meaning |
|-------|---------|
| `top1_accuracy` | Fraction of graphs where predicted best = actual best |
| `top3_accuracy` | Fraction where actual best is in top 3 predictions |
| `mean_regret` | Average % slowdown vs oracle across all graphs |
| `median_regret` | Median % slowdown (less sensitive to outliers) |
| `p95_regret` | 95th percentile slowdown (tail risk) |
| `within_5pct` | Fraction of graphs ≤ 5% slower than oracle |

**Criteria evaluated:**
- `FASTEST_REORDER` — minimize reorder time
- `FASTEST_EXECUTION` — minimize kernel execution time
- `BEST_ENDTOEND` — minimize (reorder + execution)
- `BEST_AMORTIZATION` — minimize iterations to amortize reorder cost

### 8.4 Amortization Verdicts

$$N^* = \frac{t_{\text{reorder}}}{t_{\text{baseline}} - t_{\text{reordered}}}$$

| Verdict | $N^*$ | Meaning |
|---------|------:|---------|
| **INSTANT** | < 1 | Reorder cost negligible |
| **FAST** | 1–10 | Pays off within 10 kernel runs |
| **OK** | 10–100 | Worth it for repeated analytics workloads |
| **SLOW** | > 100 | Only viable for many iterations |
| **NEVER** | ∞ | Kernel is slower after reordering — never pays off |

### 8.5 Interpreting Speedup Numbers

| Speedup Range | Interpretation |
|:---:|---|
| **1.0–1.2×** | Marginal — within noise for < 5 trials |
| **1.2–2.0×** | Solid improvement, reliable with 3+ trials |
| **2.0–5.0×** | Typical for well-matched reorderings (e.g., Leiden on social graphs) |
| **5.0–10×** | Exceptional — verify with `--graph-list <name> --trials 10` |
| **> 10×** | Suspicious — re-run with 10+ trials to confirm |

**Trial count guidelines:**

| Trials | Sufficient For |
|:------:|---|
| 2 | Crash detection only |
| 3 | Trend detection |
| 5 | Trustworthy geometric means |
| 10+ | Confirming or debunking extreme speedups |

---

## 9. Variant System

The same algorithm ID can produce different orderings depending on its
**variant** (sub-strategy).  Variants are the SSOT for "which
sub-configuration of an algorithm do we train/benchmark?"

### 9.1 Why Variants Exist

Some algorithms have multiple internal strategies that produce **different**
output orderings:

- **RabbitOrder** can build its hierarchy from a CSR scan (`csr`) or via
  Boost's community detection (`boost`).
- **GraphBrewOrder** is a multi-layer pipeline — varying presets (leiden,
  rabbit, hubcluster) and ordering strategies (hrab, tqr, hcache, streaming)
  changes the final vertex permutation.
- **GoGraphOrder** has three FEF optimization algorithms (default, fast, naive).

### 9.2 Variant Registry

Defined in `scripts/lib/core/utils.py` → `_VARIANT_ALGO_REGISTRY`:

| Algo ID | Name | Variants | Default | Why Different |
|---------|------|----------|---------|---------------|
| 8 | RabbitOrder | `csr`, `boost` | `csr` | Different hierarchy construction |
| 11 | RCM | `default`, `bnf` | `default` | Different starting vertex heuristic |
| 12 | GraphBrewOrder | `leiden`, `rabbit`, `hubcluster`, `hrab`, `tqr`, `hcache`, `streaming` | `leiden` | Different community + ordering strategies |
| 16 | GoGraphOrder | `default`, `fast`, `naive` | `default` | Different FEF optimization algorithms |

**Not in registry:** GOrder (9) has variants (`default`, `csr`, `fast`) but
they produce **equivalent** orderings (same output, different speed) — only
reorder *time* differs, not quality.  So they share a single set of weights.

### 9.3 GraphBrewOrder Strategies (7 Variants)

| Variant | Strategy | Best For |
|---------|----------|----------|
| `leiden` | Leiden partition + per-community reorder | General-purpose default |
| `rabbit` | RabbitOrder-based community hierarchy | Fast social networks |
| `hubcluster` | Hub-aware clustering | Hub-dominated graphs |
| `hrab` | Hub-Rabbit hybrid with Gorder refinement | Best cache locality overall |
| `tqr` | Cache-line-aligned tiling | Regular/mesh topologies |
| `hcache` | Hierarchical cache-aware partitioning | Deep cache hierarchies |
| `streaming` | Single-pass streaming aggregation | Fast, low-memory |

### 9.4 Expanded Configurations

When `--all-variants` is set (auto-enabled by `--target-graphs`), the pipeline
expands each varianted algorithm into all its sub-variants:

```
15 total configs when fully expanded:
  0: ORIGINAL              (baseline)
  2: SORT                  (simple)
  8: RABBITORDER_csr       (community)
  8: RABBITORDER_boost     (community)
  9: GORDER                (cache-aware)
 12: GraphBrewOrder_leiden  (multi-layer)
 12: GraphBrewOrder_rabbit
 12: GraphBrewOrder_hubcluster
 12: GraphBrewOrder_hrab
 12: GraphBrewOrder_tqr
 12: GraphBrewOrder_hcache
 12: GraphBrewOrder_streaming
 16: GOGRAPHORDER_default  (FEF)
 16: GOGRAPHORDER_fast
 16: GOGRAPHORDER_naive
```

### 9.5 Chained Orderings

Two orderings can be composed — apply the first, then the second on the
reordered graph:

| Chain | CLI Flags | Idea |
|-------|-----------|------|
| `SORT+RABBITORDER_csr` | `-o 2 -o 8:csr` | Degree-sort then community |
| `SORT+RABBITORDER_boost` | `-o 2 -o 8:boost` | Degree-sort then Boost community |
| `HUBCLUSTERDBG+RABBITORDER_csr` | `-o 7 -o 8:csr` | Hub clustering then community |
| `SORT+GraphBrewOrder_leiden` | `-o 2 -o 12:leiden:flat` | Degree-sort then Leiden pipeline |
| `DBG+GraphBrewOrder_leiden` | `-o 5 -o 12:leiden:flat` | DBG then Leiden pipeline |

---

## Quick Command Reference

```bash
# Dry-run (preview only, no execution)
python3 scripts/graphbrew_experiment.py --target-graphs 150 --size small --dry-run

# Small training run
python3 scripts/graphbrew_experiment.py --target-graphs 50 --size small

# Full training with all sizes
python3 scripts/graphbrew_experiment.py --target-graphs 150 --size all

# VLDB preview (all 8 experiments, lightweight)
python scripts/experiments/vldb_experiments_small.py --all

# VLDB full suite
python scripts/experiments/vldb_experiments.py --all --size large

# Evaluate ML models (LOGO CV)
python scripts/evaluate_all_modes.py --all --json

# Feature ablation (Experiment 4)
ADAPTIVE_ZERO_FEATURES=packing python scripts/experiments/vldb_experiments.py --exp 4

# Manual single benchmark
./bench/bin/pr -f results/graphs/email-Enron/email-Enron.sg -s -o 14 -n 3
```

---

*See also: [[AdaptiveOrder-ML]], [[Running-Benchmarks]], [[Command-Line-Reference]]*
