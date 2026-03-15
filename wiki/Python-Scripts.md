# Python Scripts Guide

Documentation for all Python tools in the GraphBrew framework.

## Overview

The scripts folder contains a single entry point, a modular library (`lib/`),
and a VLDB experiment suite (`experiments/`):

```
scripts/
├── graphbrew_experiment.py      # ⭐ MAIN: Single entry point for all experiments
├── requirements.txt             # Python dependencies
│
├── experiments/                 # 📊 VLDB 2026 paper experiment suite
│   ├── __init__.py
│   ├── vldb_paper_experiments.py  # Self-contained runner (8 experiments, auto-setup)
│   ├── vldb_generate_figures.py   # PNG figures & LaTeX table generation from JSON data
│   ├── vldb_config.py             # Shared config (graphs, algorithms, parameters)
│   ├── vldb_experiments.py        # Supplementary experiment scripts
│   ├── vldb_experiments_small.py  # Small-scale experiments
│   └── exp3_model_ablation.py     # Model ablation experiment
│
├── lib/                         # 📦 Modular library (5 sub-packages)
│   ├── __init__.py              # Re-exports every public name (backward-compatible)
│   ├── README.md                # Package map & import conventions
│   │
│   ├── core/                    # Constants, logging, data stores
│   │   ├── utils.py             # SSOT — algorithm IDs, variant registry, paths, logging
│   │   ├── graph_types.py       # Data classes (GraphInfo, BenchmarkResult, etc.)
│   │   ├── datastore.py         # BenchmarkStore, GraphPropsStore (append-only JSON DBs)
│   │   └── graph_data.py        # Graph metadata & dataset catalog
│   │
│   ├── pipeline/                # Experiment execution stages
│   │   ├── dependencies.py      # System dependency detection & installation
│   │   ├── build.py             # Binary compilation utilities
│   │   ├── download.py          # Graph downloading from SuiteSparse
│   │   ├── reorder.py           # Vertex reordering generation
│   │   ├── benchmark.py         # Performance benchmark execution
│   │   ├── cache.py             # Cache simulation analysis
│   │   ├── suitesparse_catalog.py # SuiteSparse auto-discovery (ssgetpy)
│   │   └── progress.py          # Progress tracking & reporting
│   │
│   ├── ml/                      # ML scoring & training (legacy / fallback)
│   │   ├── weights.py           # SSO scoring — PerceptronWeight.compute_score()
│   │   ├── eval_weights.py      # Weight evaluation & data loading
│   │   ├── training.py          # Iterative / batched perceptron training
│   │   ├── adaptive_emulator.py # C++ AdaptiveOrder logic emulation
│   │   ├── oracle.py            # Oracle (best-possible) analysis
│   │   └── features.py          # Graph topology feature extraction
│   │
│   ├── analysis/                # Post-run analysis & visualisation
│   │   ├── adaptive.py          # A/B testing, Leiden variant comparison
│   │   ├── metrics.py           # Amortization & end-to-end metrics
│   │   └── figures.py           # SVG / PNG plot generation
│   │
│   └── tools/                   # Standalone CLI utilities
│       ├── check_includes.py    # C++ header-include linting
│       └── regen_features.py    # Feature-vector regeneration
│
├── test/                        # Pytest suite
│   ├── __init__.py
│   ├── test_algorithm_variants.py
│   ├── test_cache_simulation.py
│   ├── test_experiment_validation.py
│   ├── test_fill_adaptive.py
│   ├── test_fill_weights_variants.py
│   ├── test_graphbrew_experiment.py
│   ├── test_multilayer_validity.py
│   ├── test_self_recording.py
│   ├── test_weight_flow.py
│   ├── data/                    # Test data fixtures
│   └── graphs/                  # Test graph fixtures
│
└── README.md
```

---

## 📊 experiments/ — Paper Experiment Suites

Two self-contained experiment runners for two papers:

### VLDB 2026 — Graph Reordering

See the [[VLDB-Experiments]] wiki page for full usage.

| Module | Purpose |
|--------|─────────|
| `vldb_paper_experiments.py` | Main runner: 8 experiments, auto-build/download/convert, output parsers |
| `vldb_generate_figures.py` | LaTeX table & PNG figure generation from experiment JSON data |
| `vldb_config.py` | Shared configuration: graph list, algorithm IDs, variant names |

### ECG/GrAPL — Cache Replacement Policies

| Module | Purpose |
|--------|─────────|
| `ecg_paper_experiments.py` | 11 experiments in 2 sections: A1-A3 accuracy validation (GRASP/P-OPT faithfulness, ECG mode equivalence), B1-B8 performance showcase (policy comparison, reorder effect, cache sweep, ECG mode comparison, fat-ID analysis) |
| `ecg_config.py` | Configuration: 9 policies, 3 ECG modes, 10 accuracy pairs, 11 reorder×policy pairs, 4 reorder variants, 6 eval graphs, cache sweep sizes |

**Section A** — Accuracy validation with PASS/FAIL reports against paper claims:
- A1: GRASP invariants (Faldu et al., HPCA'20)
- A2: P-OPT invariants (Balaji et al., HPCA'21)
- A3: ECG mode equivalence (DBG_ONLY≈GRASP, POPT_PRIMARY≈P-OPT)

**Section B** — Performance showcase:
- B1: Policy comparison, B2: Reorder effect, B3: Reorder×policy interaction
- B4: Cache size sweep, B5: Algorithm analysis, B6: Graph sensitivity
- B7: ECG mode comparison, B8: Fat-ID bit allocation

---

## ⭐ graphbrew_experiment.py - Main Orchestration

The main script provides orchestration over the `lib/` modules. It handles argument parsing and calls the appropriate phase functions.

### Quick Start

```bash
# Full pipeline: download → build → experiment → weights
python3 scripts/graphbrew_experiment.py --full --size small

# See all options
python3 scripts/graphbrew_experiment.py --help
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Graph Download** | Downloads from SuiteSparse collection (up to ~466 graphs via auto-discovery) |
| **Auto Build** | Compiles binaries if missing |
| **Memory Management** | Automatically skips graphs exceeding RAM limits |
| **Label Maps** | Pre-generates reordering maps for consistency |
| **Reordering** | Tests all 17 algorithm IDs (0-16), with variants (RCM:bnf, RabbitOrder:csr/boost, GOrder:csr/fast, GoGraphOrder:default/fast/naive, GraphBrewOrder:leiden/rabbit/hubcluster/hrab/tqr/hcache/streaming) |
| **Benchmarks** | PR, PR_SPMV, BFS, CC, CC_SV, SSSP, BC, TC |
| **Cache Simulation** | L1/L2/L3 hit rate analysis |
| **Brute-Force Validation** | Compares adaptive vs all algorithms |

---

## 🔍 Adaptive Emulator (--emulator)

**Pure Python emulator that replicates C++ AdaptiveOrder logic without recompiling.**

This is useful for:
- Analyzing how weight changes affect algorithm selection
- Testing weight configurations quickly in Python
- Understanding the two-layer selection process (type matching + perceptron)
- Debugging why a specific algorithm was chosen

### Quick Start

```bash
# Via entry point
python3 scripts/graphbrew_experiment.py --emulator

# Or directly via module (supports full argparse)
python3 -m scripts.lib.ml.adaptive_emulator --graph graphs/email-Enron/email-Enron.mtx
python3 -m scripts.lib.ml.adaptive_emulator --compare-benchmark results/benchmark_*.json
python3 -m scripts.lib.ml.adaptive_emulator --all-graphs --disable-weight w_modularity
python3 -m scripts.lib.ml.adaptive_emulator --mode best-endtoend --compare-benchmark results/benchmark.json
```

### Selection Modes

| Mode | Description |
|------|-------------|
| `fastest-reorder` | Minimize reordering time only |
| `fastest-execution` | Minimize algorithm execution time (default) |
| `best-endtoend` | Minimize (reorder_time + execution_time) |
| `best-amortization` | Minimize iterations needed to amortize reordering cost |
| `heuristic` | Feature-based heuristic (more robust) |
| `type-bench` | Type+benchmark recommendations (best accuracy) |

### How It Works

The emulator replicates the C++ AdaptiveOrder two-layer selection:

```
Layer 1: Type Matching
  - Compute graph features → normalized vector
  - Find closest type centroid (Euclidean distance)
  - OOD check: if distance > 1.5 → return ORIGINAL
  - Load that type's weights

Layer 2: Algorithm Selection
  - Compute perceptron scores for each algorithm
  - Score = bias + Σ(weight_i × feature_i)
          + quadratic cross-terms (dv×hub, mod×logN, pf×wsr)
          + convergence bonus (PR/SSSP only)
  - ORIGINAL margin check: if best - ORIGINAL < 0.05 → ORIGINAL
  - Select algorithm with highest score
```

### Emulator vs Perceptron Training

| Tool | Purpose |
|------|---------|  
| `--emulator` / `scripts.lib.ml.adaptive_emulator` | Emulate C++ selection logic, analyze weight impact |
| `--eval-weights` / `scripts.lib.ml.eval_weights` | Evaluate weights, train from benchmark data |

Use **adaptive_emulator** when you want to understand why a specific algorithm was selected.
Use **eval_weights** (via `--eval-weights`) when you want to train or evaluate weights.

---

## 📊 Weight Evaluation (--eval-weights)

**Quick evaluation script that trains weights, simulates C++ `scoreBase() × benchmarkMultiplier()` scoring, and reports accuracy/regret metrics.**

This is the fastest way to validate that your trained weights actually produce good algorithm selections without recompiling C++ or running live benchmarks.

### Quick Start

```bash
python3 scripts/graphbrew_experiment.py --eval-weights
python3 scripts/graphbrew_experiment.py --eval-weights --sg-only
```

### What It Does

1. **Loads** the latest `benchmark_*.json` and `reorder_*.json` from `results/`
2. **Trains** weights via `compute_weights_from_results()` (multi-restart perceptrons + regret-aware grid search)
3. **Saves** weights to `results/data/adaptive_models.json`
4. **Simulates** C++ scoring: for each (graph, benchmark), computes `scoreBase(algo, features) × benchmarkMultiplier(algo, bench)` for all algorithms
5. **Compares** predicted winner vs actual fastest algorithm (base-aware: variants of same algorithm count as correct)
6. **Reports** accuracy, regret, top-2 accuracy, and per-benchmark breakdown

### Output Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of (graph, benchmark) pairs where predicted base algorithm matches actual best |
| **Top-2 accuracy** | % where prediction is in the top 2 fastest algorithms |
| **Avg regret** | Average (predicted_time − best_time) / best_time across all predictions |
| **Median regret** | Median of the above (more robust to outliers) |
| **Base-aware regret** | Same as regret, but variant mismatches within the same base = 0% |
| **Per-benchmark accuracy** | Breakdown by pr, pr_spmv, bfs, cc, cc_sv, sssp, bc, tc |

### Example Output

```
=== Simulating C++ adaptive selection ===

Overall accuracy: XX/YY = ZZ.Z%
Unique predicted algorithms: N: [...]

Per-benchmark accuracy:
  bfs:      .../... = ...%
  cc:       .../... = ...%
  pr:       .../... = ...%
  sssp:     .../... = ...%
  bc:       .../... = ...%
  tc:       .../... = ...%

Average regret: X.X% (lower is better)
Top-2 accuracy: .../... = ...%
Median regret: X.X%
Base-aware avg regret: X.X% (variant mismatches = 0%)
Base-aware median regret: X.X%
```

### How C++ Scoring Is Simulated (SSO)

Scoring uses a **single source of truth** — the canonical `PerceptronWeight.compute_score()` method in `weights.py`. No duplicated formula:

```python
def _simulate_score(algo_data, feats, benchmark='pr'):
    """Simulate C++ scoreBase() × benchmarkMultiplier().

    Delegates to PerceptronWeight.compute_score() — SSO scoring.
    Covers all 17 features + 3 quadratic terms + convergence bonus
    + cache constants + benchmark multiplier.
    """
    pw = PerceptronWeight.from_dict(algo_data)
    return pw.compute_score(feats, benchmark)
```

This ensures Python evaluation matches C++ `scoreBase() × benchmarkMultiplier()` exactly, with no risk of divergence from a separately maintained formula.

### vs Other Tools

| Tool | Purpose | Updates Weights? |
|------|---------|------------------|
| `eval_weights.py` | Train + evaluate + report accuracy/regret | ✅ Yes |
| `adaptive_emulator.py` | Emulate C++ selection logic for debugging | ❌ No |

---

## ⭐ graphbrew_experiment.py - Main Orchestration (continued)

### Command-Line Options

See [[Command-Line-Reference]] for the complete CLI option reference.

Key flags: `--full` (complete pipeline), `--size small|medium|large`, `--phase reorder|benchmark|cache|weights|adaptive`, `--train` (multi-phase weight training), `--train-benchmarks` (default: pr bfs cc), `--train-iterative` (feedback loop), `--brute-force` (validation), `--auto` (auto-detect RAM/disk limits).

**Variant testing:** `--all-variants`, `--graphbrew-variants`, `--rabbit-variants`, `--gorder-variants`. Variant lists defined in `scripts/lib/core/utils.py`.

### Examples

```bash
python3 scripts/graphbrew_experiment.py --full --size small
python3 scripts/graphbrew_experiment.py --train --size small --max-graphs 5
python3 scripts/graphbrew_experiment.py --brute-force --validation-benchmark pr
python3 scripts/graphbrew_experiment.py --generate-maps --size small
```

---

## 📦 lib/ Module Reference

The `lib/` folder is organised into five sub-packages. All public names are re-exported from `lib/__init__.py` for backward compatibility.

### core/ — Constants, logging, data stores

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `utils.py` | **SSOT** constants & utilities | `ALGORITHMS`, `BENCHMARKS`, `BenchmarkResult`, variant lists, `canonical_algo_key()`, `run_command` |
| `graph_types.py` | Core types | `GraphInfo`, `AlgorithmConfig` |
| `datastore.py` | Unified data store | `BenchmarkStore`, `GraphPropsStore`, `adaptive_models.json` |
| `graph_data.py` | Per-graph storage | `GraphDataStore`, `list_graphs`, `list_runs` |

### pipeline/ — Experiment execution stages

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `dependencies.py` | System deps | `check_dependencies`, `install_dependencies` |
| `build.py` | C++ build | `build_benchmarks`, `build_converter` |
| `download.py` | Graph download | `download_graphs`, `DOWNLOAD_GRAPHS_SMALL/MEDIUM` |
| `reorder.py` | Vertex reordering | `generate_reorderings`, `ReorderResult`, `load_label_maps_index` |
| `benchmark.py` | Benchmarking | `run_benchmark`, `run_benchmarks_multi_graph`, `run_benchmarks_with_variants` |
| `cache.py` | Cache simulation | `run_cache_simulations`, `CacheResult`, `get_cache_stats_summary` |
| ~~`phases.py`~~ | _(removed)_ | Orchestration is handled directly by `graphbrew_experiment.py` |
| `progress.py` | Progress tracking | `ProgressTracker` (banners, phases, status) |

### ml/ — ML scoring & training (legacy / fallback)

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `weights.py` | **SSO** scoring (fallback) | `PerceptronWeight`, `compute_weights_from_results`, `cross_validate_logo` |
| `eval_weights.py` | Data loading | `load_all_results`, `build_performance_matrix`, `compute_graph_features` |
| `training.py` | ML training | `train_adaptive_weights_iterative`, `train_adaptive_weights_large_scale` |
| `adaptive_emulator.py` | C++ emulation | `AdaptiveOrderEmulator` |
| `oracle.py` | Oracle analysis | `compute_oracle_accuracy`, `compute_regret` |
| `features.py` | Graph features | `compute_extended_features`, `detect_graph_type`, `get_available_memory_gb` |

### analysis/ — Post-run analysis & visualisation

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `adaptive.py` | Adaptive analysis, A/B testing | `analyze_adaptive_order`, `parse_adaptive_output` |
| `metrics.py` | Amortization & E2E | `compute_amortization`, `format_amortization_table`, `AmortizationReport` |
| `figures.py` | Plot generation | SVG / PNG figures for wiki |

### tools/ — Standalone CLI utilities

| Module | Purpose |
|--------|---------|
| `check_includes.py` | Scan C++ headers for legacy includes |
| `regen_features.py` | Regenerate graph features via C++ binary |

### Key: `compute_weights_from_results()`

The primary training function: multi-restart perceptrons (5×800 epochs, z-score normalized, L2 regularized) → variant-level weight saving → regret-aware grid search for benchmark multipliers → stages to `type_0.json` (merged into `adaptive_models.json` by `export_unified_models()`). See [[Perceptron-Weights]] for details.

---

## 📏 lib/analysis/metrics.py - Amortization & End-to-End Evaluation

**Post-hoc analysis of existing benchmark results — no new benchmarks needed.**

Computes derived metrics from `benchmark_*.json` and `reorder_*.json`:
- **Amortization iterations** — how many kernel runs to break even on reorder cost
- **E2E speedup at N iterations** — speedup including amortized reorder cost
- **Head-to-head variant comparison** — crossover points between two algorithms

### Quick Start

```python
from scripts.lib.analysis.metrics import compute_amortization_report, format_amortization_table
from scripts.lib.core.datastore import BenchmarkStore

store = BenchmarkStore()
benchmark_results = store.load_benchmark_results("results/benchmark_latest.json")
reorder_results = store.load_reorder_results("results/reorder_latest.json")

report = compute_amortization_report(benchmark_results, reorder_results)
print(format_amortization_table(report))
```

### Key Formula

`iters_to_amortize = reorder_cost / (baseline_time - reordered_time)`

### Verdict Categories

| Verdict | Amort Iters | Meaning |
|---------|:-----------:|---------|
| INSTANT | < 1 | Reorder cost negligible |
| FAST | 1–10 | Pays off quickly |
| OK | 10–100 | Worth it for repeated use |
| SLOW | > 100 | Only for many iterations |
| NEVER | ∞ | Kernel is slower, never pays off |

### Output Columns

| Column | Description |
|--------|-------------|
| Kernel | Kernel-only speedup vs ORIGINAL |
| Reorder | Time spent computing the reordering |
| Amort | Iterations needed to break even |
| E2E@1 | End-to-end speedup at 1 iteration |
| E2E@10 | End-to-end speedup at 10 iterations |
| E2E@100 | End-to-end speedup at 100 iterations |
| Verdict | Human-readable break-even summary |

### Head-to-Head Comparison

The `--compare ALGO_A ALGO_B` flag produces a per-graph table showing:
- Which variant has faster kernels
- Which wins at E2E@1, @10, @100
- The **crossover iteration** where the slower-to-reorder variant overtakes
- Overall win/loss counts

The amortization report is also auto-printed by `graphbrew_experiment.py` after Phase 2 (benchmark) completes.

---

## Custom Pipeline Example

The primary entry point is `graphbrew_experiment.py`, which orchestrates all pipeline
phases directly. Use `--target-graphs N` for one-command operation:

```bash
# Full pipeline: download 100 graphs per size, benchmark, train, evaluate
python3 scripts/graphbrew_experiment.py --target-graphs 100

# Preview what would run (no execution)
python3 scripts/graphbrew_experiment.py --target-graphs 100 --dry-run

# Run individual phases
python3 scripts/graphbrew_experiment.py --phase reorder --size small
python3 scripts/graphbrew_experiment.py --phase benchmark --size small
```

See `--help` for all 90+ flags, organized into argument groups (Pipeline, Download,
Resources, Graph Selection, Algorithms, Training, Validation, Tools, etc.).

---

## Output Structure

Results are organized as:
- `results/graphs/{name}/features.json` — static graph features
- `results/logs/{name}/runs/{timestamp}/` — per-run benchmarks, reorder, weights
- `results/mappings/{name}/` — pre-generated label mappings (`.lo` + `.time`)
- `results/benchmark_*.json`, `reorder_*.json`, `cache_*.json` — aggregate results
- `results/data/adaptive_models.json` — trained model weights for C++

Use `python3 -m scripts.lib.core.graph_data --list-graphs` to browse per-graph data.

---

## Installation

```bash
cd scripts
pip install -r requirements.txt
```

### requirements.txt

```
# Core dependencies - standard library only for basic benchmarking
# These are REQUIRED for training, evaluation, and analysis scripts:
pytest>=7.0
networkx>=2.6
numpy>=1.20.0

# Optional: For extended analysis and visualization (uncomment if needed)
# pandas>=1.3.0        # For data manipulation  
# matplotlib>=3.4.0    # For plotting results
# scipy>=1.7.0         # For correlation analysis
```

---

## Troubleshooting

### Import Errors
```bash
pip install -r scripts/requirements.txt
python3 --version  # Should be 3.8+
```

### Binary Not Found
```bash
make all
make sim  # For cache simulation
```

### Permission Denied
```bash
chmod +x bench/bin/*
chmod +x bench/bin_sim/*
```

---

## Next Steps

- [[AdaptiveOrder-ML]] - ML perceptron details
- [[Running-Benchmarks]] - Command-line usage
- [[Code-Architecture]] - Codebase structure

---

[← Back to Home](Home) | [Code Architecture →](Code-Architecture)
