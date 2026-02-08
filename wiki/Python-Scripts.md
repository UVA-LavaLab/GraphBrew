# Python Scripts Guide

Documentation for all Python tools in the GraphBrew framework.

## Overview

The scripts folder contains a modular library (`lib/`) and orchestration tools:

```
scripts/
├── graphbrew_experiment.py      # ⭐ MAIN: Orchestration script (~3500 lines)
├── perceptron_experiment.py     # 🧪 ML weight experimentation (without re-running phases)
├── adaptive_emulator.py         # 🔍 C++ AdaptiveOrder logic emulation (Python)
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
| **Reordering** | Tests all 18 algorithms |
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
- `leiden`: LeidenOrder, LeidenDendrogram, LeidenCSR
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

## ⭐ graphbrew_experiment.py - Main Orchestration (continued)

### Command-Line Options

#### Dependency Management
| Option | Description |
|--------|-------------|
| `--check-deps` | Check system dependencies (g++, boost, numa, etc.) |
| `--install-deps` | Install missing system dependencies (requires sudo) |
| `--install-boost` | Download, compile, and install Boost 1.58.0 to /opt/boost_1_58_0 |

#### Pipeline Control
| Option | Description |
|--------|-------------|
| `--full` | Run complete pipeline (download → build → experiment → weights) |
| `--download-only` | Only download graphs |
| `--skip-download` | Skip graph download phase (use existing graphs) |
| `--size SIZE` | **Unified size parameter:** `small`, `medium`, `large`, `xlarge`, `all` |
| `--clean` | Clean results (keep graphs/weights) |
| `--clean-all` | Full reset for fresh start |

#### Memory Management
| Option | Description |
|--------|-------------|
| `--auto` | **Unified auto-detection:** Auto-detect both RAM and disk limits |
| `--auto-memory` | Auto-detect available RAM (uses 80% of total) |
| `--auto-disk` | Auto-detect available disk space (uses 80% of free) |
| `--max-memory GB` | Maximum RAM (GB) for graph processing |
| `--max-disk GB` | Maximum disk space (GB) for downloads |

#### Experiment Options
| Option | Description |
|--------|-------------|
| `--phase` | Run specific phase: all, reorder, benchmark, cache, weights, adaptive |
| `--quick` | Only test key algorithms (faster) |
| `--skip-cache` | Skip cache simulations |
| `--skip-expensive` | Skip BC/SSSP on large graphs |
| `--brute-force` | Run brute-force validation |

#### Algorithm Variant Testing

> **Note:** Variant lists are defined in `scripts/lib/utils.py`. Check that file for the most up-to-date list of supported variants.

| Option | Description |
|--------|-------------|
| `--all-variants` | Test ALL algorithm variants instead of just defaults |
| `--graphbrew-variants` | GraphBrewOrder clustering variants (see `GRAPHBREW_VARIANTS` in utils.py) |
| `--csr-variants` | LeidenCSR variants (see `LEIDEN_CSR_VARIANTS` in utils.py) |
| `--rabbit-variants` | RabbitOrder variants (see `RABBITORDER_VARIANTS` in utils.py) |
| `--dendrogram-variants` | LeidenDendrogram variants (see `LEIDEN_DENDROGRAM_VARIANTS` in utils.py) |
| `--vibe-variants` | VIBE variants (see `VIBE_LEIDEN_VARIANTS`, `VIBE_RABBIT_VARIANTS` in utils.py) |
| `--resolution` | Leiden resolution: `dynamic` (default, best PR), `auto`, fixed (e.g., `1.5`), `dynamic_2.0` |
| `--passes` | Leiden passes parameter (default: 3) |

**Current Default Variants:**
- **GraphBrewOrder:** `leiden` (original Leiden library)
- **LeidenCSR:** `gve` (GVE-Leiden with refinement, best modularity)
- **RabbitOrder:** `csr` (native CSR, faster, no external deps)
- **LeidenDendrogram:** `hybrid` (adaptive traversal)
- **VIBE:** `vibe` (Leiden-based, hierarchical ordering)

**LeidenCSR Variant Categories:**
| Category | Variants | Use Case |
|----------|----------|----------|
| Quality | `gve`, `gveopt`, `gveopt2`, `gveadaptive` | Best modularity/cache performance |
| Speed | `gveopt2`, `gveadaptive`, `gveturbo`, `gvefast`, `gverabbit` | Fastest reordering |
| Traversal | `dfs`, `bfs`, `hubsort` | Specific ordering patterns |
| Special | `modularity`, `gvedendo`, `gveoptdendo` | Modularity-optimized, dendrogram-based |
| **VIBE** | `vibe`, `vibe:dfs`, `vibe:dbg`, `vibe:streaming`, `vibe:lazyupdate`, `vibe:dynamic` | Leiden + configurable ordering |
| **VIBE Rabbit** | `vibe:rabbit`, `vibe:rabbit:dfs`, `vibe:rabbit:dbg` | RabbitOrder + post-ordering |

**VIBE Variants (Unified Framework):**
| Variant | Algorithm | Description |
|---------|-----------|-------------|
| `vibe` | Leiden | Multi-pass community detection + hierarchical ordering |
| `vibe:dfs` | Leiden | + DFS dendrogram traversal |
| `vibe:dbg` | Leiden | + DBG within each community |
| `vibe:streaming` | Leiden | + Lazy aggregation (faster) |
| `vibe:lazyupdate` | Leiden | + Batched community weight updates (reduces atomics) |
| `vibe:dynamic` | Leiden | + Per-pass resolution adjustment |
| `vibe:rabbit` | RabbitOrder | Single-pass parallel aggregation |
| `vibe:rabbit:dfs` | RabbitOrder | + DFS post-ordering |
| `vibe:rabbit:dbg` | RabbitOrder | + DBG post-ordering |

**VIBE Resolution Modes:**
| Mode | Option | Description |
|------|--------|-------------|
| Auto | `vibe:auto` | Graph-adaptive resolution (computed once, default) |
| Dynamic | `vibe:dynamic` | Adjusted per-pass based on runtime metrics |
| Fixed | `vibe:0.75` | User-specified fixed value |

**Example - Compare GVE variants on 5 largest graphs:**
```bash
python3 scripts/graphbrew_experiment.py \
  --phase cache \
  --graph-list wiki-topcats cit-Patents as-Skitter web-BerkStan web-Google \
  --csr-variants gve gveopt gveopt2 gveadaptive \
  --rabbit-variants csr boost \
  --benchmarks pr bfs cc sssp \
  --skip-build --auto
```

**Example - Test VIBE variants:**
```bash
python3 scripts/graphbrew_experiment.py \
  --phase benchmark \
  --graph-list web-Google wiki-Talk \
  --csr-variants vibe vibe:dfs vibe:dbg vibe:dynamic vibe:rabbit vibe:rabbit:dfs \
  --benchmarks pr bfs \
  --skip-build --auto
```

#### Label Mapping (Consistent Reordering)
| Option | Description |
|--------|-------------|
| `--precompute` | Pre-generate and use label maps |
| `--generate-maps` | Pre-generate .lo mapping files |
| `--use-maps` | Use pre-generated label maps |

#### Training Options
| Option | Description |
|--------|-------------|
| `--train` | Complete training pipeline: reorder → benchmark → cache sim → update weights |
| `--train-iterative` | Run iterative training feedback loop |
| `--train-batched` | Run large-scale batched training |
| `--target-accuracy` | Target accuracy % (default: 80) |

#### Deprecated Parameters

| Deprecated | Use Instead |
|------------|-------------|
| `--graphs SIZE` | `--size SIZE` |
| `--download-size SIZE` | `--size SIZE` |
| `--auto-memory --auto-disk` | `--auto` |
| `--key-only` | `--quick` |
| `--fill-weights` | `--train` |
| `--train-adaptive` | `--train-iterative` |
| `--train-large` | `--train-batched` |

### Examples

```bash
# One-click full experiment
python3 scripts/graphbrew_experiment.py --full --size small

# Quick test with key algorithms
python3 scripts/graphbrew_experiment.py --size small --quick

# Pre-generate label maps
python3 scripts/graphbrew_experiment.py --generate-maps --size small

# Train: complete pipeline (cache sim, weights, everything)
python3 scripts/graphbrew_experiment.py --train --size small --max-graphs 5

# Skip download phase (use existing graphs)
python3 scripts/graphbrew_experiment.py --full --size large --skip-download

# Clean and start fresh
python3 scripts/graphbrew_experiment.py --clean-all --full --size small
```

---

## 📦 lib/ Module Reference

The `lib/` folder contains modular, reusable components. Each module can be used independently or via the phase orchestration system.

### lib/graph_types.py - Data Classes

Central type definitions used across all modules:

```python
from scripts.lib.graph_types import GraphInfo, BenchmarkResult, CacheResult, ReorderResult

# GraphInfo - Graph metadata
GraphInfo(name="web-Stanford", path="graphs/web-Stanford/web-Stanford.mtx", 
          size_mb=5.2, nodes=281903, edges=2312497)

# BenchmarkResult - Benchmark execution result
BenchmarkResult(graph="web-Stanford", algorithm_id=7, algorithm_name="HUBCLUSTERDBG",
                benchmark="pr", avg_time=0.234, speedup=1.45, success=True)

# CacheResult - Cache simulation result
CacheResult(graph="web-Stanford", algorithm_id=7, algorithm_name="HUBCLUSTERDBG",
            benchmark="pr", l1_miss_rate=0.12, l2_miss_rate=0.08, l3_miss_rate=0.02)

# ReorderResult - Reordering result
ReorderResult(graph="web-Stanford", algorithm_id=7, algorithm_name="HUBCLUSTERDBG",
              time_seconds=1.23, mapping_file="mappings/web-Stanford/HUBCLUSTERDBG.lo")
```

### lib/phases.py - Phase Orchestration

High-level phase functions for building custom pipelines:

```python
from scripts.lib.phases import (
    PhaseConfig,
    run_reorder_phase,
    run_benchmark_phase,
    run_cache_phase,
    run_weights_phase,
    run_full_pipeline,
)

# Create configuration
config = PhaseConfig(
    benchmarks=['pr', 'bfs', 'cc'],
    trials=3,
    skip_slow=True
)

# Run individual phases
reorder_results, label_maps = run_reorder_phase(graphs, algorithms, config)
benchmark_results = run_benchmark_phase(graphs, algorithms, label_maps, config)

# Or run full pipeline
results = run_full_pipeline(graphs, algorithms, config, phases=['reorder', 'benchmark'])
```

### lib/utils.py - Core Utilities

**Single Source of Truth** for all shared constants. Never duplicate these elsewhere:

```python
from scripts.lib.utils import (
    # Algorithm definitions
    ALGORITHMS,          # {0: "ORIGINAL", 1: "RANDOM", ..., 17: "LeidenCSR"}
    SLOW_ALGORITHMS,     # {9, 10, 11} - Gorder, Corder, RCM
    BENCHMARKS,          # ['pr', 'bfs', 'cc', 'sssp', 'bc', 'tc']
    
    # Variant lists
    LEIDEN_CSR_VARIANTS, GRAPHBREW_VARIANTS,
    RABBITORDER_VARIANTS, LEIDEN_DENDROGRAM_VARIANTS,
    VIBE_LEIDEN_VARIANTS,   # ['vibe', 'vibe:dfs', 'vibe:dbg', ..., 'vibe:conn', 'vibe:hrab']
    VIBE_RABBIT_VARIANTS,   # ['vibe:rabbit', 'vibe:rabbit:dfs', ...]
    
    # Size thresholds (MB)
    SIZE_SMALL, SIZE_MEDIUM, SIZE_LARGE, SIZE_XLARGE,
    
    # Timeout constants (seconds)
    TIMEOUT_REORDER,     # 43200 (12 hours)
    TIMEOUT_BENCHMARK,   # 600 (10 min)
    TIMEOUT_SIM,         # 1200 (20 min)
    TIMEOUT_SIM_HEAVY,   # 3600 (1 hour)
    
    # Utilities
    run_command,         # Execute shell commands
    get_timestamp,       # Formatted timestamps
)
```

### lib/features.py - Graph Features

Graph feature computation and system utilities:

```python
from scripts.lib.features import (
    # Graph type detection
    detect_graph_type,
    compute_extended_features,
    
    # System utilities
    get_available_memory_gb,
    get_num_threads,
    estimate_graph_memory_gb,
)

# Compute graph features
features = compute_extended_features("graph.mtx")
# Returns: {modularity, density, avg_degree, degree_variance, clustering_coefficient, ...}

# Detect graph type
graph_type = detect_graph_type(features)  # "social", "web", "road", etc.
```

### lib/dependencies.py - System Dependencies

Automatic system dependency detection and installation:

```python
from scripts.lib.dependencies import (
    check_dependencies,      # Check all required dependencies
    install_dependencies,    # Install missing dependencies (needs sudo)
    install_boost_158,       # Download and compile Boost 1.58.0
    check_boost_158,         # Check if Boost 1.58.0 is installed
    detect_platform,         # Detect OS and package manager
    get_package_manager,     # Get system package manager commands
)

# Check dependencies
status = check_dependencies()
# Returns dict with: g++, boost, numa, tcmalloc, python versions and status

# Install missing dependencies (requires sudo)
install_dependencies()

# Install Boost 1.58.0 for RabbitOrder
install_boost_158()  # Downloads, compiles with bootstrap/b2, installs to /opt/boost_1_58_0

# Check Boost 1.58 specifically
version = check_boost_158()  # Returns version string or None
```

### lib/download.py - Graph Downloading

Download graphs from SuiteSparse:

```python
from scripts.lib.download import (
    DOWNLOAD_GRAPHS_SMALL,   # 16 small graphs
    DOWNLOAD_GRAPHS_MEDIUM,  # 28 medium graphs
    download_graphs,
    get_catalog_stats,
)

# Download small graphs
download_graphs(DOWNLOAD_GRAPHS_SMALL, output_dir="./graphs")

# Get catalog statistics
stats = get_catalog_stats()
print(f"Total graphs: {stats['total']}, Total size: {stats['total_size_gb']:.1f} GB")
```

### lib/reorder.py - Reordering

Generate vertex reorderings:

```python
from scripts.lib.reorder import (
    generate_reorderings,
    generate_reorderings_with_variants,
    load_label_maps_index,
)

# Generate reorderings for all algorithms
results = generate_reorderings(graphs, algorithms, bin_dir="bench/bin")

# Load existing label maps
label_maps = load_label_maps_index("results")
```

### lib/benchmark.py - Benchmarking

Run performance benchmarks:

```python
from scripts.lib.benchmark import (
    run_benchmark,
    run_benchmark_suite,
    parse_benchmark_output,
)

# Run single benchmark
result = run_benchmark(graph_path, algorithm_id, benchmark="pr", bin_dir="bench/bin")

# Run full suite
results = run_benchmark_suite(graphs, algorithms, benchmarks=['pr', 'bfs'])
```

### lib/cache.py - Cache Simulation

Run cache simulations:

```python
from scripts.lib.cache import (
    run_cache_simulations,
    get_cache_stats_summary,
)

# Run simulations
results = run_cache_simulations(graphs, algorithms, benchmarks=['pr'])

# Get summary statistics
summary = get_cache_stats_summary(results)
```

### lib/weights.py - Weight Management

Type-based weight management for AdaptiveOrder:

```python
from scripts.lib.weights import (
    assign_graph_type,
    update_type_weights_incremental,
    get_best_algorithm_for_type,
    load_type_registry,
    cross_validate_logo,              # NEW: Leave-One-Graph-Out validation
    compute_weights_from_results,     # Correlation-based weight computation
)

# Assign graph to a type based on features
type_name, is_new = assign_graph_type("web-Stanford", features)

# Update weights incrementally (with L2 regularization)
update_type_weights_incremental(type_name, algorithm_name, benchmark, speedup)

# Get best algorithm for a type
best_algo = get_best_algorithm_for_type(type_name, benchmark="pr")

# Cross-validate with Leave-One-Graph-Out
result = cross_validate_logo(benchmark_results, graph_features, type_registry)
print(f"LOGO accuracy: {result['accuracy']:.1%}")
print(f"Overfitting score: {result['overfitting_score']:.2f}")
```

**PerceptronWeight dataclass** fields (all used in scoring):
- Core: `bias`, `w_modularity`, `w_log_nodes`, `w_log_edges`, `w_density`, `w_avg_degree`, `w_degree_variance`, `w_hub_concentration`
- Extended: `w_clustering_coeff`, `w_avg_path_length`, `w_diameter`, `w_community_count`
- New graph-aware: `w_packing_factor`, `w_forward_edge_fraction`, `w_working_set_ratio`
- Quadratic: `w_dv_x_hub`, `w_mod_x_logn`, `w_pf_x_wsr`
- Convergence: `w_fef_convergence` (PR/SSSP only)
- Cache: `cache_l1_impact`, `cache_l2_impact`, `cache_l3_impact`, `cache_dram_penalty`
- Time: `w_reorder_time`

**Training features:**
- L2 regularization (`WEIGHT_DECAY = 1e-4`) prevents weight explosion
- ORIGINAL is trained as a regular algorithm (no longer skipped)
- `_metadata.avg_reorder_time` is calibrated from actual measurements

### lib/training.py - ML Training

Train adaptive weights:

```python
from scripts.lib.training import (
    train_adaptive_weights_iterative,
    train_adaptive_weights_large_scale,
)

# Iterative training
result = train_adaptive_weights_iterative(
    graphs=graphs,
    bin_dir="bench/bin",
    target_accuracy=0.85,
    max_iterations=10
)
print(f"Final accuracy: {result.final_accuracy:.2%}")
```

### lib/analysis.py - Adaptive Analysis

Analyze adaptive ordering:

```python
from scripts.lib.analysis import (
    analyze_adaptive_order,
    compare_adaptive_vs_fixed,
    run_subcommunity_brute_force,
)

# Analyze adaptive ordering
results = analyze_adaptive_order(graphs, bin_dir="bench/bin")

# Compare adaptive vs fixed algorithms
comparison = compare_adaptive_vs_fixed(graphs, fixed_algorithms=[7, 15, 16])
```

### lib/progress.py - Progress Tracking

Visual progress tracking:

```python
from scripts.lib.progress import ProgressTracker

progress = ProgressTracker()
progress.banner("EXPERIMENT", "Running GraphBrew benchmarks")
progress.phase_start("REORDERING", "Generating vertex reorderings")
progress.info("Processing graph: web-Stanford")
progress.success("Completed 10/15 graphs")
progress.phase_end("Reordering complete")
```

### lib/graph_data.py - Per-Graph Data Storage

Organized storage and retrieval of per-graph experiment data:

```python
from scripts.lib.graph_data import (
    GraphDataStore,
    list_all_graphs,
    list_runs_for_graph,
    get_latest_run,
)

# Initialize data store
store = GraphDataStore("results")

# Save features for a graph
store.save_features("web-Stanford", {
    "nodes": 281903,
    "edges": 2312497,
    "modularity": 0.45,
    "degree_variance": 1.8,
})

# Save benchmark result for a run
store.save_benchmark_result("web-Stanford", run_timestamp, "pr", "HUBCLUSTERDBG", {
    "avg_time": 0.234,
    "speedup": 1.45,
})

# Get all data for a graph
all_data = store.get_graph_data("web-Stanford")

# List all graphs with data
graphs = list_all_graphs("results")

# List runs for a specific graph
runs = list_runs_for_graph("results", "web-Stanford")
```

**CLI Usage:**
```bash
# List all graphs
python3 -m scripts.lib.graph_data --list-graphs

# Show graph details
python3 -m scripts.lib.graph_data --show-graph email-Enron

# Export to CSV
python3 -m scripts.lib.graph_data --export-csv results/all_data.csv

# List runs for a graph
python3 -m scripts.lib.graph_data --list-runs email-Enron

# Show run details
python3 -m scripts.lib.graph_data --show-run email-Enron 20260127_145547
```

### lib/results.py - Result File I/O

Read and write result files:

```python
from scripts.lib.results import (
    save_results,
    load_results,
    find_latest_results,
)

# Save results with timestamp
save_results(benchmark_results, "results", "benchmark")

# Load latest results
results = find_latest_results("results", "benchmark")
```

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
algorithms = [0, 7, 15, 16]  # ORIGINAL, HUBCLUSTERDBG, LeidenOrder, LeidenDendrogram

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

GraphBrew separates **static graph features** from **run-specific experiment data**:

```
results/
├── graphs/                   # Static per-graph features
│   └── {graph_name}/
│       └── features.json     # Graph topology (nodes, edges, modularity, etc.)
│
├── logs/                     # Run-specific data and command logs
│   └── {graph_name}/
│       ├── runs/             # Timestamped experiment runs
│       │   └── {timestamp}/
│       │       ├── benchmarks/   # Per-algorithm benchmark results
│       │       ├── reorder/      # Reorder times and mapping info
│       │       ├── weights/      # Computed perceptron weights
│       │       └── summary.json  # Run metadata
│       ├── reorder_*.log         # Individual reorder command outputs
│       ├── benchmark_*.log       # Individual benchmark outputs
│       └── cache_*.log           # Individual cache sim outputs
│
├── mappings/                 # Pre-generated label mappings
│   ├── index.json            # Mapping index
│   └── {graph_name}/         # Per-graph mappings
│       ├── HUBCLUSTERDBG.lo  # Label order file
│       └── HUBCLUSTERDBG.time # Reorder timing
│
├── reorder_*.json            # Aggregate reorder results
├── benchmark_*.json          # Aggregate benchmark results
└── cache_*.json              # Aggregate cache simulation results

scripts/weights/              # Type-based weights
├── active/                   # C++ reads from here
│   ├── type_registry.json    # Graph → type mapping
│   ├── type_0.json           # Cluster 0 weights
│   └── type_N.json           # Additional clusters
├── merged/                   # Accumulated from all runs
└── runs/                     # Historical snapshots
```

### Managing Experiment Runs

Use `graph_data.py` CLI to manage per-graph experiment data:

```bash
# List all runs for a graph
python3 -m scripts.lib.graph_data --list-runs ca-GrQc

# Show details of a specific run
python3 -m scripts.lib.graph_data --show-run ca-GrQc 20260127_152449

# Clean up old runs (keep last 5)
python3 -m scripts.lib.graph_data --cleanup-runs --max-runs 5

# Migrate old data structure to new
python3 -m scripts.lib.graph_data --migrate
```

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
