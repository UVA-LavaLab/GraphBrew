# Python Scripts Guide

Documentation for all Python tools in the GraphBrew framework.

## Overview

```
scripts/
├── graphbrew_experiment.py      # ⭐ MAIN: One-click unified experiment pipeline
├── requirements.txt             # Python dependencies
├── perceptron_weights.json      # Trained ML weights (auto-generated)
│
├── analysis/                    # Utility libraries
│   ├── correlation_analysis.py  # Feature-algorithm correlation functions
│   └── perceptron_features.py   # ML feature extraction utilities
│
├── benchmark/                   # Specialized benchmarks
│   └── run_pagerank_convergence.py  # PageRank iteration analysis
│
├── download/                    # Graph acquisition
│   └── download_graphs.py       # Standalone graph downloader
│
└── utils/                       # Shared utilities
    └── common.py                # ALGORITHMS dict, parsing helpers
```

---

## ⭐ graphbrew_experiment.py - Unified Pipeline (Recommended)

The main script that combines all functionality into a single one-click solution.

### Quick Start

```bash
# Full pipeline: download → build → experiment → weights
python3 scripts/graphbrew_experiment.py --full --download-size SMALL

# See all options
python3 scripts/graphbrew_experiment.py --help
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Graph Download** | Downloads from SuiteSparse collection (96 graphs available) |
| **Auto Build** | Compiles binaries if missing |
| **Memory Management** | Automatically skips graphs exceeding RAM limits |
| **Label Maps** | Pre-generates reordering maps for consistency |
| **Reordering** | Tests all 20 algorithms |
| **Benchmarks** | PR, BFS, CC, SSSP, BC |
| **Cache Simulation** | L1/L2/L3 hit rate analysis |
| **Perceptron Training** | Generates weights for AdaptiveOrder |
| **Brute-Force Validation** | Compares adaptive vs all algorithms |

### Command-Line Options

#### Pipeline Control
| Option | Description |
|--------|-------------|
| `--full` | Run complete pipeline (download → build → experiment → weights) |
| `--download-only` | Only download graphs |
| `--download-size` | SMALL (16), MEDIUM (34), LARGE (40), XLARGE (6), ALL (96 graphs) |
| `--clean` | Clean results (keep graphs/weights) |
| `--clean-all` | Full reset for fresh start |

#### Memory Management
| Option | Description |
|--------|-------------|
| `--max-memory GB` | Maximum RAM (GB) for graph processing. Graphs exceeding this limit are skipped. |
| `--auto-memory` | Automatically detect available RAM and skip graphs that won't fit (uses 80% of total) |

#### Disk Space Management
| Option | Description |
|--------|-------------|
| `--max-disk GB` | Maximum disk space (GB) for downloads. Downloads stop when cumulative size exceeds limit. |
| `--auto-disk` | Automatically limit downloads to available disk space (uses 80% of free space) |

#### Experiment Options
| Option | Description |
|--------|-------------|
| `--phase` | Run specific phase: all, reorder, benchmark, cache, weights, adaptive |
| `--graphs` | Graph size: all, small, medium, large, custom |
| `--key-only` | Only test key algorithms (faster) |
| `--skip-cache` | Skip cache simulations |
| `--brute-force` | Run brute-force validation |

#### Label Mapping (Consistent Reordering)
| Option | Description |
|--------|-------------|
| `--generate-maps` | Pre-generate .lo mapping files for consistent reordering |
| `--use-maps` | Use pre-generated label maps (avoids regenerating each run) |

#### Iterative Training (Adaptive Weight Optimization)
| Option | Description |
|--------|-------------|
| `--train-adaptive` | Run iterative training feedback loop |
| `--train-large` | Run large-scale training with batching and multi-benchmark support |
| `--target-accuracy` | Target accuracy % for training (default: 80) |
| `--max-iterations` | Maximum training iterations (default: 10) |
| `--learning-rate` | Weight adjustment rate (default: 0.1) |
| `--batch-size` | Batch size for large-scale training (default: 8) |
| `--train-benchmarks` | Benchmarks for multi-benchmark training (default: pr bfs cc) |
| `--init-weights` | Initialize/upgrade weights file with enhanced features |
| `--fill-weights` | Fill ALL weight fields: runs cache sim, graph features, benchmark analysis |

> **Note:** `--full` automatically enables `--generate-maps` and `--use-maps` for consistent results across all benchmarks.

### Examples

```bash
# One-click full experiment
python3 scripts/graphbrew_experiment.py --full --download-size SMALL

# Download medium graphs only
python3 scripts/graphbrew_experiment.py --download-only --download-size MEDIUM

# Download all graphs that fit in 32GB RAM
python3 scripts/graphbrew_experiment.py --download-only --download-size ALL --max-memory 32

# Limit downloads to 50GB disk space
python3 scripts/graphbrew_experiment.py --download-only --download-size ALL --max-disk 50

# Auto-detect RAM and disk space limits
python3 scripts/graphbrew_experiment.py --full --download-size ALL --auto-memory --auto-disk

# Run experiment on existing graphs
python3 scripts/graphbrew_experiment.py --phase all --graphs small

# Quick test with key algorithms
python3 scripts/graphbrew_experiment.py --graphs small --key-only

# Run brute-force validation
python3 scripts/graphbrew_experiment.py --brute-force

# Generate weights from existing results
python3 scripts/graphbrew_experiment.py --phase weights

# Pre-generate label maps (also records reorder times)
python3 scripts/graphbrew_experiment.py --generate-maps --graphs small

# Iterative training to reach 90% accuracy
python3 scripts/graphbrew_experiment.py --train-adaptive --target-accuracy 90 --graphs small

# Full training with custom learning rate
python3 scripts/graphbrew_experiment.py --train-adaptive --target-accuracy 85 --learning-rate 0.05

# Clean and start fresh
python3 scripts/graphbrew_experiment.py --clean-all --full --download-size SMALL

# Fill ALL weight fields (cache impacts, topology features, benchmark weights)
python3 scripts/graphbrew_experiment.py --fill-weights --graphs small --max-graphs 5

# Fill weights on all graphs within memory and disk limits
python3 scripts/graphbrew_experiment.py --fill-weights --auto-memory --auto-disk --download-size ALL
```

### Output Structure

All outputs go to `./results/`:

```
results/
├── graphs/                   # Downloaded graphs (if using --full)
│   └── {graph_name}/         # Each graph in its own directory
│       └── {graph_name}.mtx  # Matrix Market format
├── mappings/                 # Pre-generated label mappings
│   ├── index.json            # Mapping index (graph → algo → path)
│   └── {graph_name}/         # Per-graph mappings
│       ├── HUBCLUSTERDBG.lo  # Label order for each algorithm
│       ├── HUBCLUSTERDBG.time # Reorder time for this algorithm
│       ├── LeidenHybrid.lo
│       └── ...
├── reorder_times_*.json      # All reorder timings (per graph/algo)
├── reorder_times_*.csv       # CSV version for analysis
├── benchmark_*.json          # Benchmark execution results
├── cache_*.json              # Cache simulation results (L1/L2/L3 hit rates)
├── perceptron_weights.json   # Trained ML weights with metadata
├── perceptron_weights_*.json # Timestamped weight backups (auto-generated)
├── brute_force_*.json        # Validation results
├── training_*/               # Iterative training output (if --train-adaptive)
│   ├── training_summary.json # Overall training results
│   ├── weights_iter1.json    # Weights after each iteration
│   ├── best_weights_*.json   # Best weights (highest accuracy)
│   └── brute_force_*.json    # Per-iteration analysis
└── logs/                     # Execution logs
```

---

## Data Classes

The script uses these dataclasses for structured data:

### GraphInfo
```python
@dataclass
class GraphInfo:
    name: str           # Graph name
    path: str           # Path to graph file
    size_mb: float      # File size in MB
    is_symmetric: bool  # Whether graph is symmetric
    nodes: int          # Number of vertices
    edges: int          # Number of edges
```

### ReorderResult
```python
@dataclass
class ReorderResult:
    graph: str              # Graph name
    algorithm_id: int       # Algorithm ID (0-20)
    algorithm_name: str     # Algorithm name
    reorder_time: float     # Time to reorder (seconds)
    mapping_file: str       # Path to .lo file (if generated)
    success: bool           # Whether reordering succeeded
    error: str              # Error message if failed
```

### BenchmarkResult
```python
@dataclass
class BenchmarkResult:
    graph: str              # Graph name
    algorithm_id: int       # Algorithm ID
    algorithm_name: str     # Algorithm name
    benchmark: str          # Benchmark name (pr, bfs, etc.)
    trial_time: float       # Average trial time
    speedup: float          # Speedup vs ORIGINAL
    nodes: int              # Graph nodes
    edges: int              # Graph edges
    success: bool           # Whether benchmark succeeded
    error: str              # Error message if failed
```

### CacheResult
```python
@dataclass
class CacheResult:
    graph: str              # Graph name
    algorithm_id: int       # Algorithm ID
    algorithm_name: str     # Algorithm name
    benchmark: str          # Benchmark name
    l1_hit_rate: float      # L1 cache hit rate (%)
    l2_hit_rate: float      # L2 cache hit rate (%)
    l3_hit_rate: float      # L3 cache hit rate (%)
    success: bool           # Whether simulation succeeded
    error: str              # Error message if failed
```

### PerceptronWeight
```python
@dataclass
class PerceptronWeight:
    # Core weights
    bias: float = 0.5               # Base preference (0.0-1.0+)
    w_modularity: float = 0.0       # Weight for modularity
    w_log_nodes: float = 0.0        # Weight for log(nodes)
    w_log_edges: float = 0.0        # Weight for log(edges)
    w_density: float = 0.0          # Weight for edge density
    w_avg_degree: float = 0.0       # Weight for average degree
    w_degree_variance: float = 0.0  # Weight for degree variance
    w_hub_concentration: float = 0.0 # Weight for hub concentration
    
    # Cache impact weights
    cache_l1_impact: float = 0.0    # L1 cache hit rate impact
    cache_l2_impact: float = 0.0    # L2 cache hit rate impact
    cache_l3_impact: float = 0.0    # L3 cache hit rate impact
    cache_dram_penalty: float = 0.0 # DRAM access penalty
    
    # Reorder time weight
    w_reorder_time: float = 0.0     # Penalty for reorder time
    
    # Extended graph structure features (NEW)
    w_clustering_coeff: float = 0.0   # Local clustering coefficient effect
    w_avg_path_length: float = 0.0    # Average path length sensitivity
    w_diameter: float = 0.0           # Diameter effect
    w_community_count: float = 0.0    # Sub-community count effect
    
    # Per-benchmark weight adjustments (NEW)
    benchmark_weights: Dict[str, float] = field(default_factory=lambda: {
        'pr': 1.0, 'bfs': 1.0, 'cc': 1.0, 'sssp': 1.0, 'bc': 1.0
    })
    
    # Training metadata
    _metadata: Dict = field(default_factory=dict)  # win_rate, avg_speedup, sample_count
```

### DownloadableGraph
```python
@dataclass
class DownloadableGraph:
    name: str           # Graph name
    url: str            # Download URL
    size_mb: int        # Expected size in MB
    nodes: int          # Number of nodes
    edges: int          # Number of edges
    symmetric: bool     # Whether graph is symmetric
    category: str       # Category (social, web, road, etc.)
    description: str    # Human-readable description
```

---

## Installation

### Requirements

```bash
cd scripts
pip install -r requirements.txt
```

### requirements.txt Contents

```
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
scipy>=1.6.0
networkx>=2.5
tqdm>=4.50.0
```

---

## Utility Scripts

### download/download_graphs.py

Standalone graph downloader (also integrated into main script).

```bash
# List available graphs
python3 scripts/download/download_graphs.py --list

# Download specific size category
python3 scripts/download/download_graphs.py --size MEDIUM --output-dir ./graphs
```

### benchmark/run_pagerank_convergence.py

Specialized PageRank convergence analysis.

```bash
# Analyze convergence iterations by reordering
python3 scripts/benchmark/run_pagerank_convergence.py --graphs-dir ./graphs
```

### utils/common.py

Shared utilities imported by other scripts:
- `ALGORITHMS` dictionary mapping IDs to names
- Output parsing functions
- Graph path utilities

---

## Library Modules

### analysis/correlation_analysis.py

Functions for computing feature-algorithm correlations.

```python
from scripts.analysis.correlation_analysis import compute_graph_features

# Compute features for a graph
features = compute_graph_features('graphs/my_graph.el')
print(f"Modularity: {features['modularity']:.4f}")
print(f"Density: {features['density']:.6f}")
```

### analysis/perceptron_features.py

ML feature extraction for perceptron training.

```python
from scripts.analysis.perceptron_features import extract_features

# Extract ML features
features = extract_features(graph_path)
```

---

## Common Tasks

### Task 1: Full Experiment from Scratch

```bash
# One command does everything
python3 scripts/graphbrew_experiment.py --full --download-size MEDIUM
```

### Task 2: Quick Validation

```bash
# Test on small graphs with key algorithms
python3 scripts/graphbrew_experiment.py --graphs small --key-only --skip-cache
```

### Task 3: Regenerate Weights

```bash
# After running experiments, regenerate weights
python3 scripts/graphbrew_experiment.py --phase weights
```

### Task 4: Brute-Force Algorithm Comparison

```bash
# Test adaptive vs all 20 algorithms
python3 scripts/graphbrew_experiment.py --brute-force --graphs small
```

---

## Troubleshooting

### Import Errors

```bash
# Install missing packages
pip install -r scripts/requirements.txt

# Check Python version
python3 --version  # Should be 3.6+
```

### Binary Not Found

```bash
# The unified script auto-builds, but manually:
make all
make sim  # For cache simulation
```

### Permission Denied

```bash
chmod +x bench/bin/*
chmod +x bench/bin_sim/*
```

### Download Failures

```bash
# Retry with force flag
python3 scripts/graphbrew_experiment.py --download-only --force-download
```

---

## Next Steps

- [[AdaptiveOrder-ML]] - ML perceptron details
- [[Running-Benchmarks]] - Command-line usage
- [[Benchmark-Suite]] - Automated experiments

---

[← Back to Home](Home) | [AdaptiveOrder ML →](AdaptiveOrder-ML)
