# Python Scripts Guide

Documentation for all Python tools in the GraphBrew framework.

## Overview

The scripts folder contains a modular library (`lib/`) and orchestration tools:

```
scripts/
├── graphbrew_experiment.py      # ⭐ MAIN: Orchestration script (~3500 lines)
├── perceptron_experiment.py     # 🧪 ML weight experimentation (without re-running phases)
├── adaptive_emulator.py         # 🔍 C++ AdaptiveOrder logic emulation (Python)
├── eval_weights.py              # 📊 Weight evaluation: train → simulate C++ scoring → report accuracy
├── analyze_metrics.py           # 📏 Amortization & end-to-end evaluation from result JSONs
├── requirements.txt             # Python dependencies
│
├── lib/                         # 📦 Modular library (~14300 lines total)
│   ├── __init__.py              # Module exports
│   ├── graph_types.py                 # Data classes (GraphInfo, BenchmarkResult, etc.)
│   ├── phases.py                # Phase orchestration (run_reorder_phase, etc.)
│   ├── utils.py                 # Core utilities (ALGORITHMS, run_command, etc.)
│   ├── features.py              # Graph feature computation & system utilities
│   ├── dependencies.py          # System dependency detection & installation
│   ├── download.py              # Graph downloading from SuiteSparse
│   ├── build.py                 # Binary compilation utilities
│   ├── reorder.py               # Vertex reordering generation
│   ├── benchmark.py             # Performance benchmark execution
│   ├── cache.py                 # Cache simulation analysis
│   ├── weights.py               # Type-based weight management
│   ├── weight_merger.py         # Cross-run weight consolidation
│   ├── training.py              # ML weight training
│   ├── analysis.py              # Adaptive order analysis
│   ├── graph_data.py            # Per-graph data storage & retrieval
│   ├── progress.py              # Progress tracking & reporting
│   └── results.py               # Result file I/O
│
├── test/                        # Pytest suite
│   ├── test_weight_flow.py      # Weight generation/loading tests
│   ├── test_weight_merger.py    # Merger consolidation tests
│   ├── test_fill_adaptive.py    # Fill-weights pipeline tests
│   ├── test_cache_simulation.py # Cache simulation tests
│   ├── test_graphbrew_experiment.py # Main experiment tests
│   └── graphs/                  # Test graph fixtures
│
├── weights/                     # Type-based weight files
│   ├── active/                  # C++ reads from here (working copy)
│   │   ├── type_registry.json   # Maps graphs → types + centroids
│   │   ├── type_0.json          # Cluster 0 weights
│   │   └── type_N.json          # Additional clusters
│   ├── merged/                  # Accumulated from all runs
│   └── runs/                    # Historical snapshots
│
├── examples/                    # Example scripts
│   ├── batch_process.py         # Batch processing example
│   ├── compare_algorithms.py    # Algorithm comparison example
│   ├── custom_pipeline.py       # Custom phase-based pipeline example
│   └── quick_test.py            # Quick testing example
└── requirements.txt             # Python dependencies (optional)
```

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
| **Graph Download** | Downloads from SuiteSparse collection (87 graphs available) |
| **Auto Build** | Compiles binaries if missing |
| **Memory Management** | Automatically skips graphs exceeding RAM limits |
| **Label Maps** | Pre-generates reordering maps for consistency |
| **Reordering** | Tests all 17 algorithms |
| **Benchmarks** | PR, BFS, CC, SSSP, BC, TC |
| **Cache Simulation** | L1/L2/L3 hit rate analysis |
| **Perceptron Training** | Generates weights for AdaptiveOrder |
| **Brute-Force Validation** | Compares adaptive vs all algorithms |

---

## 🧪 perceptron_experiment.py - ML Experimentation

**Experiment with perceptron configurations WITHOUT re-running expensive phases.**

This script loads existing benchmark results and lets you:
- Try different weight training methods (speedup, winrate, rank, hybrid)
- Run grid search to find optimal configurations
- Interactively tweak weights and evaluate accuracy
- Export optimized weights to active directory for C++ to use

### Quick Start

```bash
# Show current weights and accuracy
python3 scripts/perceptron_experiment.py --show

# Run grid search to find best configuration
python3 scripts/perceptron_experiment.py --grid-search

# Train with specific method and export
python3 scripts/perceptron_experiment.py --train --method hybrid --export

# Interactive mode for manual tuning
python3 scripts/perceptron_experiment.py --interactive
```

### Training Methods

| Method | Description |
|--------|-------------|
| `speedup` | Bias = average speedup over ORIGINAL baseline |
| `winrate` | Bias = win rate (how often algorithm is best) |
| `rank` | Bias = inverse average rank across benchmarks |
| `hybrid` | Weighted combination: 0.4×speedup + 0.4×winrate + 0.2×rank |
| `per_benchmark` | Benchmark-specific multipliers (generates `benchmark_weights` per algorithm) |

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--show` | Show current weights and evaluate accuracy |
| `--analyze` | Taxonomy analysis: best algorithms per category per benchmark |
| `--grid-search` | Run grid search over 32 configurations |
| `--train` | Train new weights with specified method |
| `--method METHOD` | Training method: speedup, winrate, rank, hybrid, per_benchmark |
| `--scale SCALE` | Bias scale factor (default: 1.0) |
| `--clusters N` | Number of graph clusters for type-based weights (default: 1) |
| `--benchmark BENCH` | Benchmark to evaluate (default: pr) |
| `--export` | Export weights to `scripts/weights/active/` |
| `--interactive` | Enter interactive mode for manual tuning |
| `--save-results FILE` | Save experiment results to JSON file |

### Taxonomy Analysis (--analyze)

The `--analyze` command provides insights into which algorithms work best for different graph types and benchmarks:

```bash
python3 scripts/perceptron_experiment.py --analyze
```

**Output includes:**
- **Algorithm Taxonomy:** Categorizes algorithms into groups (basic, hub, community, leiden, composite)
- **Graph Type Detection:** Identifies graph type (social, web, road, citation, p2p, email, random)
- **Best Algorithm per Category:** Shows which algorithm from each category performs best per benchmark
- **Overall Winners:** Which algorithm wins most often for each graph type

**Algorithm Categories:**
- `basic`: ORIGINAL, RANDOM, SORT
- `hub`: HUBSORT, HUBCLUSTER, DBG, HUBSORTDBG, HUBCLUSTERDBG
- `community`: GORDER, RABBITORDER, CORDER, RCM
- `leiden`: LeidenOrder, LeidenCSR
- `composite`: AdaptiveOrder, GraphBrewOrder

### Example: Reproducible Experimentation

```bash
# 1. Run expensive phases once
python3 scripts/graphbrew_experiment.py --full --size medium --auto

# 2. Experiment with different perceptron configs (fast, no re-running)
python3 scripts/perceptron_experiment.py --grid-search

# 3. Analyze which algorithms work best per benchmark/graph type
python3 scripts/perceptron_experiment.py --analyze

# 4. Train with per-benchmark weights
python3 scripts/perceptron_experiment.py --train --method per_benchmark --export

# 5. Validate with AdaptiveOrder
./bench/bin/pr -f graph.el -s -o 14 -n 3
```

---

## 🔍 adaptive_emulator.py - C++ Logic Emulation

**Pure Python emulator that replicates C++ AdaptiveOrder logic without recompiling.**

This is useful for:
- Analyzing how weight changes affect algorithm selection
- Testing weight configurations quickly in Python
- Understanding the two-layer selection process (type matching + perceptron)
- Debugging why a specific algorithm was chosen

### Quick Start

```bash
# Emulate for a single graph
python3 scripts/adaptive_emulator.py --graph graphs/email-Enron/email-Enron.mtx

# Compare emulation vs actual benchmark results
python3 scripts/adaptive_emulator.py --compare-benchmark results/benchmark_*.json

# Disable a weight to see its impact
python3 scripts/adaptive_emulator.py --all-graphs --disable-weight w_modularity

# Different selection modes
python3 scripts/adaptive_emulator.py --mode best-endtoend --compare-benchmark results/benchmark.json
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

### vs perceptron_experiment.py

| Tool | Purpose |
|------|---------|
| `adaptive_emulator.py` | Emulate C++ selection logic, analyze weight impact |
| `perceptron_experiment.py` | Train new weights from benchmark data |

Use **adaptive_emulator.py** when you want to understand why a specific algorithm was selected.
Use **perceptron_experiment.py** when you want to train better weights.

---

## 📊 eval_weights.py - Weight Evaluation & C++ Scoring Simulation

**Quick evaluation script that trains weights, simulates C++ `scoreBase() × benchmarkMultiplier()` scoring, and reports accuracy/regret metrics.**

This is the fastest way to validate that your trained weights actually produce good algorithm selections without recompiling C++ or running live benchmarks.

### Quick Start

```bash
python3 scripts/eval_weights.py
```

### What It Does

1. **Loads** the latest `benchmark_*.json` and `reorder_*.json` from `results/`
2. **Trains** weights via `compute_weights_from_results()` (multi-restart perceptrons + regret-aware grid search)
3. **Saves** weights to `scripts/weights/active/type_0.json`
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
| **Per-benchmark accuracy** | Breakdown by pr, bfs, cc, sssp |

### Example Output

```
=== Simulating C++ adaptive selection ===

Overall accuracy: 88/188 = 46.8%
Unique predicted algorithms: 13: ['DBG', 'GORDER', 'HUBCLUSTER', ...]

Per-benchmark accuracy:
  bfs: 25/47 = 53.2%
  cc:  20/47 = 42.6%
  pr:  22/47 = 46.8%
  sssp: 21/47 = 44.7%

Average regret: 10.1% (lower is better)
Top-2 accuracy: 122/188 = 64.9%
Median regret: 5.6%
Base-aware avg regret: 6.2% (variant mismatches = 0%)
Base-aware median regret: 2.6%
```

### How C++ Scoring Is Simulated

```python
def simulate_score(algo_data, feats, bench_type):
    """Mimic C++ scoreBase() * benchmarkMultiplier()"""
    s = algo_data['bias']
    s += algo_data['w_modularity'] * feats['modularity']
    s += algo_data['w_log_nodes'] * feats['log_nodes']
    s += algo_data['w_log_edges'] * feats['log_edges']
    s += algo_data['w_density'] * feats['density']
    s += algo_data['w_avg_degree'] * feats['avg_degree'] / 100.0
    s += algo_data['w_degree_variance'] * feats['degree_variance']
    s += algo_data['w_hub_concentration'] * feats['hub_concentration']
    s += algo_data['w_clustering_coeff'] * feats['clustering_coefficient']
    # ... (all features)
    
    # Benchmark multiplier from regret-aware grid search
    mult = algo_data['benchmark_weights'][bench_type]
    return s * mult
```

### vs Other Tools

| Tool | Purpose | Updates Weights? |
|------|---------|------------------|
| `eval_weights.py` | Train + evaluate + report accuracy/regret | ✅ Yes |
| `adaptive_emulator.py` | Emulate C++ selection logic for debugging | ❌ No |
| `perceptron_experiment.py` | Grid search over training configurations | ✅ Yes |

---

## ⭐ graphbrew_experiment.py - Main Orchestration (continued)

### Command-Line Options

See [[Command-Line-Reference]] for the complete CLI option reference.

Key flags: `--full` (complete pipeline), `--size small|medium|large`, `--phase reorder|benchmark|cache|weights`, `--train` (6-phase weight training), `--train-iterative` (feedback loop), `--brute-force` (validation), `--auto` (auto-detect RAM/disk limits).

**Variant testing:** `--all-variants`, `--csr-variants`, `--graphbrew-variants`, `--rabbit-variants`, `--dendrogram-variants`, `--graphbrew-variants`. Variant lists defined in `scripts/lib/utils.py`.

### Examples

```bash
python3 scripts/graphbrew_experiment.py --full --size small
python3 scripts/graphbrew_experiment.py --train --size small --max-graphs 5
python3 scripts/graphbrew_experiment.py --brute-force --validation-benchmark pr
python3 scripts/graphbrew_experiment.py --generate-maps --size small
```

---

## 📦 lib/ Module Reference

The `lib/` folder (~14,300 lines) contains modular, reusable components:

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `graph_types.py` | Data classes | `GraphInfo`, `BenchmarkResult`, `CacheResult`, `ReorderResult` |
| `phases.py` | Phase orchestration | `PhaseConfig`, `run_reorder_phase`, `run_benchmark_phase`, `run_full_pipeline` |
| `utils.py` | Constants & utilities | `ALGORITHMS`, `BENCHMARKS`, variant lists, `run_command` |
| `features.py` | Graph features | `compute_extended_features`, `detect_graph_type`, `get_available_memory_gb` |
| `dependencies.py` | System deps | `check_dependencies`, `install_dependencies`, `install_boost_158` |
| `download.py` | Graph download | `download_graphs`, `DOWNLOAD_GRAPHS_SMALL/MEDIUM` |
| `reorder.py` | Vertex reordering | `generate_reorderings`, `load_label_maps_index` |
| `benchmark.py` | Benchmarking | `run_benchmark`, `run_benchmark_suite` |
| `cache.py` | Cache simulation | `run_cache_simulations`, `get_cache_stats_summary` |
| `weights.py` | Weight management | `compute_weights_from_results`, `cross_validate_logo`, `assign_graph_type` |
| `weight_merger.py` | Cross-run merge | Weight consolidation across training runs |
| `training.py` | ML training | `train_adaptive_weights_iterative`, `train_adaptive_weights_large_scale` |
| `analysis.py` | Adaptive analysis | `analyze_adaptive_order`, `parse_adaptive_output` |
| `graph_data.py` | Per-graph storage | `GraphDataStore`, `list_all_graphs`, `list_runs_for_graph` |
| `progress.py` | Progress tracking | `ProgressTracker` (banners, phases, status) |
| `results.py` | Result I/O | `save_results`, `load_results`, `find_latest_results` |
| `metrics.py` | Amortization & E2E | `compute_amortization`, `compute_variant_comparison`, `AmortizationReport` |

### Key: `compute_weights_from_results()`

The primary training function: multi-restart perceptrons (5×800 epochs, z-score normalized, L2 regularized) → variant pre-collapse → regret-aware grid search for benchmark multipliers → saves `type_0.json`. See [[Perceptron-Weights]] for details.

---

## 📏 analyze_metrics.py - Amortization & End-to-End Evaluation

**Post-hoc analysis of existing benchmark results — no new benchmarks needed.**

Computes derived metrics from `benchmark_*.json` and `reorder_*.json`:
- **Amortization iterations** — how many kernel runs to break even on reorder cost
- **E2E speedup at N iterations** — speedup including amortized reorder cost
- **Head-to-head variant comparison** — crossover points between two algorithms

### Quick Start

```bash
# Full amortization report from latest results
python3 scripts/analyze_metrics.py --results-dir results/

# Filter to specific benchmarks/graphs
python3 scripts/analyze_metrics.py --results-dir results/ \
  --benchmarks pr bfs --graphs web-Google soc-Slashdot0902

# Head-to-head: RabbitOrder vs GraphBrew:hrab
python3 scripts/analyze_metrics.py --results-dir results/ \
  --compare RABBITORDER_csr GraphBrewOrder_graphbrew:hrab

# JSON output for scripting
python3 scripts/analyze_metrics.py --results-dir results/ --json
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

Create custom experiment pipelines using `lib/phases.py`:

```python
#!/usr/bin/env python3
"""Custom GraphBrew pipeline example."""

import sys
sys.path.insert(0, "scripts")

from lib.phases import PhaseConfig, run_reorder_phase, run_benchmark_phase
from lib.graph_types import GraphInfo
from lib.progress import ProgressTracker

# Discover graphs
graphs = [
    GraphInfo(name="web-Stanford", path="graphs/web-Stanford/web-Stanford.mtx",
              size_mb=5.2, nodes=281903, edges=2312497)
]

# Select algorithms
algorithms = [0, 7, 15, 16]  # ORIGINAL, HUBCLUSTERDBG, LeidenOrder, LeidenCSR

# Create configuration
config = PhaseConfig(
    benchmarks=['pr', 'bfs'],
    trials=3,
    progress=ProgressTracker()
)

# Run phases
reorder_results, label_maps = run_reorder_phase(graphs, algorithms, config)
benchmark_results = run_benchmark_phase(graphs, algorithms, label_maps, config)

# Print results
for r in benchmark_results:
    if r.success:
        print(f"{r.graph} / {r.algorithm_name} / {r.benchmark}: {r.avg_time:.4f}s")
```

See `scripts/examples/custom_pipeline.py` for a complete example.

---

## Output Structure

Results are organized as:
- `results/graphs/{name}/features.json` — static graph features
- `results/logs/{name}/runs/{timestamp}/` — per-run benchmarks, reorder, weights
- `results/mappings/{name}/` — pre-generated label mappings (`.lo` + `.time`)
- `results/benchmark_*.json`, `reorder_*.json`, `cache_*.json` — aggregate results
- `scripts/weights/active/` — type-based weights for C++

Use `python3 -m scripts.lib.graph_data --list-graphs` to browse per-graph data.

---

## Installation

```bash
cd scripts
pip install -r requirements.txt
```

### requirements.txt

```
# Core dependencies - NONE REQUIRED
# All benchmark scripts use only Python 3.8+ standard library

# Optional: For extended analysis and visualization (uncomment if needed)
# numpy>=1.20.0        # For statistical analysis
# pandas>=1.3.0        # For data manipulation  
# matplotlib>=3.4.0    # For plotting results
# scipy>=1.7.0         # For correlation analysis
# networkx>=2.6        # For graph analysis
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
