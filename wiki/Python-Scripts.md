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
| **Graph Download** | Downloads from SuiteSparse collection |
| **Auto Build** | Compiles binaries if missing |
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
| `--download-size` | SMALL (4 graphs), MEDIUM (11), LARGE (2), ALL (17) |
| `--clean` | Clean results (keep graphs/weights) |
| `--clean-all` | Full reset for fresh start |

#### Experiment Options
| Option | Description |
|--------|-------------|
| `--phase` | Run specific phase: all, reorder, benchmark, cache, weights, adaptive |
| `--graphs` | Graph size: all, small, medium, large, custom |
| `--key-only` | Only test key algorithms (faster) |
| `--skip-cache` | Skip cache simulations |
| `--brute-force` | Run brute-force validation |

### Examples

```bash
# One-click full experiment
python3 scripts/graphbrew_experiment.py --full --download-size SMALL

# Download medium graphs only
python3 scripts/graphbrew_experiment.py --download-only --download-size MEDIUM

# Run experiment on existing graphs
python3 scripts/graphbrew_experiment.py --phase all --graphs small

# Quick test with key algorithms
python3 scripts/graphbrew_experiment.py --graphs small --key-only

# Run brute-force validation
python3 scripts/graphbrew_experiment.py --brute-force

# Generate weights from existing results
python3 scripts/graphbrew_experiment.py --phase weights

# Clean and start fresh
python3 scripts/graphbrew_experiment.py --clean-all --full --download-size SMALL
```

### Output Structure

All outputs go to `./results/`:

```
results/
├── graphs/                   # Downloaded graphs (if using --full)
├── mappings/                 # Reordering label maps
├── reorder_*.json            # Reordering times
├── benchmark_*.json          # Benchmark results
├── cache_*.json              # Cache simulation results
├── perceptron_weights.json   # Trained ML weights
├── brute_force_*.json        # Validation results
└── logs/                     # Execution logs
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
