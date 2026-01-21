[![Build Status](https://app.travis-ci.com/UVA-LavaLab/GraphBrew.svg?branch=main)](https://app.travis-ci.com/UVA-LavaLab/GraphBrew)
[![Wiki](https://img.shields.io/badge/📚_Wiki-Documentation-blue?style=flat)](https://github.com/UVA-LavaLab/GraphBrew/wiki)
[<p align="center"><img src="./docs/figures/logo.svg" width="200" ></p>](#graphbrew)

# GraphBrew <img src="./docs/figures/logo_left.png" width="50" align="center"> GAPBS

This repository contains the GAP Benchmarks Suite [(GAPBS)](https://github.com/sbeamer/gapbs), modified to reorder graphs and improve cache performance on various graph algorithms.

> **📖 New to GraphBrew?** Check out our comprehensive **[Wiki Documentation](https://github.com/UVA-LavaLab/GraphBrew/wiki)** for detailed guides on installation, algorithms, and usage!

## Enhancements with Cache Friendly Graphs (Graph Brewing)

* **GraphBrew:** Graph reordering (multi-layered) for improved cache performance. See **[GraphBrewOrder Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/GraphBrewOrder)**.
* **Leiden Order:** [link](https://github.com/puzzlef/leiden-communities-openmp) Community clustering order with Louvian/refinement step. See **[Community Detection Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Community-Detection)**.
* **Rabbit Order:** [link](https://github.com/araij/rabbit_order) Community clustering order with incremental aggregation.
* **Degree-Based Grouping:** [link](https://github.com/ease-lab/dbg) Implementing degree-based grouping strategies to test benchmark performance.
* **Gorder:** [link](https://github.com/datourat/Gorder) Window based ordering with reverse Cuthill-McKee (RCM) algorithm.
* **Corder:** [link](https://github.com/yuang-chen/Corder-TPDS-21) Workload Balancing via Graph Reordering on Multicore Systems.
* **Leiden Dendrogram Variants:** New algorithms that exploit community hierarchy for optimal node ordering.
* **AdaptiveOrder:** ML-based perceptron selector that automatically chooses the best algorithm. See **[AdaptiveOrder ML Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/AdaptiveOrder-ML)**.

<!-- * **P-OPT Segmentation:** [link](https://github.com/CMUAbstract/POPT-CacheSim-HPCA21) Exploring graph caching techniques for efficient handling of large-scale graphs.
* **GraphIt-DSL:** [link](https://github.com/GraphIt-DSL/graphit) Integration of GraphIt-DSL segment graphs to improve locality. -->

## Algorithm Selection Guide

Choosing the right reordering algorithm depends on your graph characteristics. For detailed algorithm descriptions, see the **[Reordering Algorithms Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Reordering-Algorithms)**.

| Graph Type | Recommended Algorithm | Rationale |
|------------|----------------------|-----------|
| **Social Networks** (high clustering) | `LeidenHybrid (20)` or `AdaptiveOrder (15)` | Hub-aware DFS exploits community structure |
| **Web Graphs** (power-law degree) | `LeidenDFSHub (17)` or `HubClusterDBG (7)` | Prioritizes high-degree hubs for cache efficiency |
| **Road Networks** (low clustering) | `RCM (11)` or `Gorder (9)` | BFS-based approaches work well for sparse graphs |
| **Unknown/Mixed** | `AdaptiveOrder (15)` | Let the ML perceptron choose automatically |

### Quick Start Examples
```bash
# Use AdaptiveOrder for automatic best selection
./bench/bin/bfs -g 20 -o 15

# Use LeidenHybrid (best overall for social/web graphs)
./bench/bin/pr -f graph.mtx -o 20

# Use GraphBrew with LeidenDFSHub for per-community ordering
./bench/bin/pr -f graph.mtx -o 13:10:17

# Chain multiple orderings (Leiden then Sort refinement)
./bench/bin/bfs -f graph.mtx -o 20 -o 2
```

> 📖 **More examples?** See the **[Quick Start Guide](https://github.com/UVA-LavaLab/GraphBrew/wiki/Quick-Start)** and **[Command Line Reference](https://github.com/UVA-LavaLab/GraphBrew/wiki/Command-Line-Reference)** in the wiki.

## Segmentation for Scalable Graph Processing
* **Cagra:** [link1](https://github.com/CMUAbstract/POPT-CacheSim-HPCA21)/[link2](https://github.com/GraphIt-DSL/graphit) Integration of P-OPT/GraphIt-DSL segment graphs to improve locality.
* **Trust:** [link](https://github.com/wzbxpy/TRUST) Graph partition for Triangle counting on large graph.

## GAP Benchmarks

This project contains a collection of Graph Analytics for Performance [(GAPBS)](https://github.com/sbeamer/gapbs) benchmarks implemented in C++. The benchmarks are designed to exercise the performance of graph algorithms on a CPU. For implementation details, see **[Graph Benchmarks Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Graph-Benchmarks)**.

**Key Algorithms**

* **bc:** Betweenness Centrality 
* **bfs:** Breadth-First Search (Direction Optimized) 
* **cc:** Connected Components (Afforest)
* **cc_sv:** Connected Components (ShiloachVishkin)
* **pr:** PageRank
* **pr_spmv:** PageRank (using sparse matrix-vector multiplication)
* **sssp:**  Single-Source Shortest Paths
* **tc:** Triangle Counting

## 🚀 One-Click Experiment Pipeline

Run the complete GraphBrew experiment workflow with a single command:

```bash
# Clone and run - that's it!
git clone https://github.com/UVA-LavaLab/GraphBrew.git
cd GraphBrew
python3 scripts/graphbrew_experiment.py --full --download-size SMALL
```

This single command will:
1. **Download** benchmark graphs from SuiteSparse collection (96 graphs available)
2. **Build** the benchmark binaries automatically
3. **Generate** label mappings for consistent reordering across all benchmarks
4. **Run** performance benchmarks (BFS, PR, CC, SSSP, BC) with all 20 algorithms
5. **Execute** cache simulations for L1/L2/L3 hit rate analysis
6. **Train** perceptron weights for AdaptiveOrder (with cache + reorder time features)

All results are saved to `./results/` for easy analysis.

### Available Options

```bash
# Download graphs only
python3 scripts/graphbrew_experiment.py --download-only --download-size MEDIUM

# Run experiment on existing graphs
python3 scripts/graphbrew_experiment.py --phase all

# Quick test with key algorithms
python3 scripts/graphbrew_experiment.py --graphs small --key-only

# Run brute-force validation (test adaptive vs all 20 algorithms)
python3 scripts/graphbrew_experiment.py --brute-force

# Use pre-generated label maps for consistent reordering
python3 scripts/graphbrew_experiment.py --generate-maps --use-maps

# Auto-detect RAM and disk limits (skips graphs that won't fit)
python3 scripts/graphbrew_experiment.py --full --download-size ALL --auto-memory --auto-disk

# Set explicit memory/disk limits
python3 scripts/graphbrew_experiment.py --full --download-size ALL --max-memory 32 --max-disk 100

# Clean and start fresh
python3 scripts/graphbrew_experiment.py --clean-all --full --download-size SMALL

# See all options
python3 scripts/graphbrew_experiment.py --help
```

### Graph Catalog

| Size | Graphs | Total Size | Categories |
|------|--------|------------|------------|
| `SMALL` | 16 | ~62 MB | communication, collaboration, p2p, social, citation |
| `MEDIUM` | 34 | ~1.1 GB | web, road, commerce, mesh, synthetic, infrastructure |
| `LARGE` | 40 | ~27 GB | social, web, collaboration, road, mesh, synthetic |
| `XLARGE` | 6 | ~63 GB | massive web (twitter7, webbase-2001), Kronecker |
| `ALL` | **96** | ~92 GB | Complete benchmark set |

### Resource Management

| Option | Description |
|--------|-------------|
| `--max-memory GB` | Skip graphs requiring more than this RAM |
| `--auto-memory` | Auto-detect RAM, use 80% as limit |
| `--max-disk GB` | Limit total download size |
| `--auto-disk` | Auto-detect disk space, use 80% as limit |

## Prerequisites

Before you begin, ensure you have the following installed on your system, [(section)](#installing-prerequisites). For detailed installation steps, see **[Installation Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Installation)**.

- **Ubuntu**: All testing has been done on Ubuntu `22.04+` Operating System.
- **GCC**: The GNU Compiler Collection, specifically `g++9` which supports C++11 or later.
- **Make**: The build utility to automate the compilation.
- **OpenMP**: Support for parallel programming in C++.
### Compile with RabbitOrder
   * Go to Makefile <[line:8](https://github.com/atmughrabi/GraphBrew/blob/main/Makefile#L8)> make sure `RABBIT_ENABLE = 1`
```bash
# make RABBIT_ENABLE=1 // disables RabbitOrder dependencies
<OR>
make RABBIT_ENABLE=1
```
   * **Boost** C++ library (1.58.0).
   * **libnuma** (2.0.9).
   * **libtcmalloc\_minimal** in google-perftools (2.1).

# GraphBrew Analysis Scripts

The `scripts/` directory contains Python tools for comprehensive benchmarking and analysis. For detailed usage, see **[Python Scripts Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Python-Scripts)**.

**Main Script (Unified Pipeline):**
```
scripts/
├── graphbrew_experiment.py    # ⭐ One-click unified experiment pipeline
│                              #    - Downloads graphs from SuiteSparse
│                              #    - Builds binaries automatically  
│                              #    - Runs all benchmarks & simulations
│                              #    - Generates perceptron weights
│                              #    - Supports brute-force validation
├── requirements.txt           # Python dependencies
└── perceptron_weights.json    # ML weights (auto-generated)
```

**Utility Scripts:**
```
scripts/
├── download/
│   └── download_graphs.py          # Standalone graph downloader
├── benchmark/
│   └── run_pagerank_convergence.py # PageRank convergence analysis
├── analysis/
│   ├── correlation_analysis.py     # Feature-algorithm correlation library
│   └── perceptron_features.py      # ML feature extraction utilities
└── utils/
    └── common.py                   # Shared utilities (ALGORITHMS dict)
```

## Results Directory Structure

Benchmark results are organized in the `results/` folder:

```
results/
├── graphs/                    # Downloaded graphs (if using --full)
│   └── {graph_name}/          # Each graph in its own directory
│       └── {name}.mtx         # Matrix Market format
├── mappings/                  # Pre-generated label orderings
│   ├── index.json             # Mapping index: graph → algo → path
│   └── {graph_name}/          # Per-graph mappings
│       ├── HUBCLUSTERDBG.lo   # Label order file for each algorithm
│       ├── LeidenHybrid.lo
│       └── ...
├── reorder_*.json             # Reordering times per algorithm
├── benchmark_*.json           # Benchmark execution results  
├── cache_*.json               # Cache simulation results (L1/L2/L3)
├── perceptron_weights.json    # Trained ML weights with metadata
├── brute_force_*.json         # Validation results
└── logs/                      # Execution logs
```

### Perceptron Weights Format

The trained weights include cache impact, reorder time, and benchmark-specific multipliers:

```json
{
  "LeidenHybrid": {
    "bias": 0.85,
    "w_modularity": 0.25,
    "w_log_nodes": 0.1,
    "w_log_edges": 0.1,
    "w_density": -0.05,
    "w_avg_degree": 0.15,
    "w_degree_variance": 0.15,
    "w_hub_concentration": 0.25,
    "cache_l1_impact": 0.1,
    "cache_l2_impact": 0.05,
    "cache_l3_impact": 0.02,
    "cache_dram_penalty": -0.1,
    "w_reorder_time": -0.0078,
    "w_clustering_coeff": 0.0,
    "w_avg_path_length": 0.0,
    "w_diameter": 0.0,
    "w_community_count": 0.0,
    "benchmark_weights": {
      "pr": 1.0,
      "bfs": 1.0,
      "cc": 1.0,
      "sssp": 1.0,
      "bc": 1.0
    },
    "_metadata": {
      "win_rate": 0.85,
      "avg_speedup": 2.34,
      "times_best": 42,
      "sample_count": 50,
      "avg_reorder_time": 1.234,
      "avg_l1_hit_rate": 85.2,
      "avg_l2_hit_rate": 92.1,
      "avg_l3_hit_rate": 98.5
    }
  }
}
```

**Weight Categories:**

| Category | Weights | Description |
|----------|---------|-------------|
| **Core** | `bias`, `w_modularity`, `w_log_nodes`, `w_log_edges`, `w_density`, `w_avg_degree`, `w_degree_variance`, `w_hub_concentration` | Basic graph feature weights |
| **Extended** | `w_clustering_coeff`, `w_avg_path_length`, `w_diameter`, `w_community_count` | Advanced graph topology features |
| **Cache** | `cache_l1_impact`, `cache_l2_impact`, `cache_l3_impact`, `cache_dram_penalty` | Cache performance weights (from simulation) |
| **Time** | `w_reorder_time` | Penalty for slow reordering algorithms |
| **Benchmark** | `benchmark_weights: {pr, bfs, cc, sssp, bc}` | Multipliers for benchmark-specific tuning |

> 📖 **Understanding results?** See **[Correlation Analysis Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Correlation-Analysis)** and **[AdaptiveOrder ML Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/AdaptiveOrder-ML)** for interpretation guides.

---

# 🔬 Cache Simulation Framework

GraphBrew includes a detailed cache simulation framework for analyzing memory access patterns and cache behavior across reordering algorithms.

## Building Cache Simulation

```bash
# Build all simulation binaries
make all-sim

# Build specific algorithm simulation
make sim-pr   # PageRank
make sim-bfs  # BFS
make sim-cc   # Connected Components

# Clean simulation binaries
make clean-sim
```

## Running Cache Simulations

```bash
# Basic simulation with default Intel Xeon cache config
./bench/bin_sim/pr -g 18 -o 12 -n 1

# Export statistics to JSON
CACHE_OUTPUT_JSON=cache_stats.json ./bench/bin_sim/pr -g 18 -o 12 -n 1

# Custom cache configuration
CACHE_L1_SIZE=32768 CACHE_L1_WAYS=8 CACHE_L1_POLICY=LRU \
CACHE_L2_SIZE=262144 CACHE_L3_SIZE=12582912 \
./bench/bin_sim/bfs -g 16 -o 7 -n 1
```

## Cache Benchmark Suite

The unified experiment script includes comprehensive cache analysis:

```bash
# One-click: includes cache simulation
python3 scripts/graphbrew_experiment.py --full --download-size SMALL

# Cache simulation only
python3 scripts/graphbrew_experiment.py --phase cache --graphs small

# Skip heavy simulations (BC, SSSP) on large graphs
python3 scripts/graphbrew_experiment.py --phase cache --skip-heavy
```

Results are saved to `results/cache_*.json` with L1/L2/L3 hit rates for each graph/algorithm combination.

## Cache Features for ML

Cache performance metrics are integrated into perceptron weights:

| Weight | Description |
|--------|-------------|
| `cache_l1_impact` | Bonus for algorithms with high L1 hit rate |
| `cache_l2_impact` | Bonus for algorithms with high L2 hit rate |
| `cache_l3_impact` | Bonus for algorithms with high L3 hit rate |
| `cache_dram_penalty` | Penalty for DRAM accesses (cache misses) |

Metadata includes average hit rates:
- `avg_l1_hit_rate`: Mean L1 hit rate across benchmarks
- `avg_l2_hit_rate`: Mean L2 hit rate across benchmarks
- `avg_l3_hit_rate`: Mean L3 hit rate across benchmarks

For detailed documentation, see **[Cache Simulation Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Cache-Simulation)**.

---

# 🔬 Reproducing Our Experiments

This section provides step-by-step instructions for researchers to reproduce our benchmarking results.

## Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/atmughrabi/GraphBrew.git
cd GraphBrew

# Install system dependencies (Ubuntu 22.04+)
sudo apt-get update
sudo apt-get install -y build-essential g++-9 libboost-all-dev libnuma-dev google-perftools

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt

# Build all benchmarks with RabbitOrder support
make clean && make RABBIT_ENABLE=1 all
```

## Step 2: Download Benchmark Graphs

```bash
# List available graphs with sizes
python3 scripts/download/download_graphs.py --list

# Download MEDIUM graphs (~600MB, good for testing)
python3 scripts/download/download_graphs.py --size MEDIUM --output-dir ./graphs

# Download ALL graphs including LARGE (~72GB, for full experiments)
python3 scripts/download/download_graphs.py --size ALL --output-dir ./graphs

# Validate downloaded graphs
python3 scripts/download/download_graphs.py --validate
```

**Available Graph Sizes:**
| Size | Graphs | Download | Use Case |
|------|--------|----------|----------|
| SMALL | 16 | ~62 MB | Quick testing |
| MEDIUM | 34 | ~1.1 GB | Development & validation |
| LARGE | 40 | ~27 GB | Full paper experiments |
| XLARGE | 6 | ~63 GB | Massive-scale testing |
| ALL | 96 | ~91 GB | Complete benchmark |

## Step 3: Run Benchmarks

### Quick Validation (5-10 minutes)
```bash
# One-click full pipeline (recommended)
python3 scripts/graphbrew_experiment.py --full --download-size SMALL

# Quick test with key algorithms
python3 scripts/graphbrew_experiment.py --graphs small --key-only --skip-cache
```

### Full Benchmark Suite (several hours)
```bash
# Run all algorithms on all graphs with cache simulation
python3 scripts/graphbrew_experiment.py --full --download-size MEDIUM

# Run with automatic memory/disk filtering
python3 scripts/graphbrew_experiment.py --full --download-size ALL --auto-memory --auto-disk

# Or run benchmarks separately from downloads
python3 scripts/graphbrew_experiment.py --phase benchmark --graphs medium --trials 16
```

### Multi-Source Benchmarks (BFS, SSSP, BC)
For traversal algorithms, the benchmark automatically runs multiple source nodes:
```bash
python3 scripts/graphbrew_experiment.py --phase benchmark --benchmarks bfs sssp bc --trials 16
```

## Step 4: Analyze Results

### Generate Perceptron Weights (AdaptiveOrder)
```bash
# Generate weights from existing benchmark + cache results
python3 scripts/graphbrew_experiment.py --phase weights
```

### Brute-Force Validation
```bash
# Compare adaptive selection vs all 20 algorithms
python3 scripts/graphbrew_experiment.py --brute-force --graphs small
```

### Results Location
All results are saved to `./results/`:
- `benchmark_*.json` - Execution times and speedups
- `cache_*.json` - L1/L2/L3 cache hit rates
- `reorder_*.json` - Reordering times per algorithm
- `perceptron_weights.json` - Trained ML weights with metadata

# Quick test with synthetic graphs (uses defaults for untested algorithms)
python3 scripts/analysis/correlation_analysis.py --quick

# Or specify a custom output location:
python3 scripts/analysis/correlation_analysis.py \
    --graphs-dir ./graphs \
    --weights-file ./custom_weights.json
```

> 📖 **Understanding perceptron weights?** See **[AdaptiveOrder ML Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/AdaptiveOrder-ML)** and **[Perceptron Weights Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Perceptron-Weights)**.

## Step 5: Verify Correctness

```bash
# Quick topology test (ensures reordering preserves graph structure)
make test-topology

# Full verification with all algorithms
make test-topology-full
```

---

## 📊 Expected Results Summary

Based on our experiments, you should observe:

| Graph Type | Best Algorithm | Typical Speedup |
|------------|---------------|-----------------|
| Social Networks | LeidenHybrid (20) | 1.2-2.5x |
| Web Graphs | LeidenDFSHub (17) | 1.3-2.0x |
| Road Networks | RCM (11) / Gorder (9) | 1.1-1.5x |
| Citation Networks | HubClusterDBG (7) | 1.2-1.8x |
| Mixed/Unknown | AdaptiveOrder (15) | Near-optimal |

> 📖 **Need help?** Check out the **[FAQ](https://github.com/UVA-LavaLab/GraphBrew/wiki/FAQ)** and **[Troubleshooting Guide](https://github.com/UVA-LavaLab/GraphBrew/wiki/Troubleshooting)** in the wiki.

---

# Legacy Experiment Configuration

> **Note:** The sections below describe the legacy experiment system. For new experiments, we recommend using the Python scripts described above.

## Manual Graph Download (Alternative)

If you prefer to download graphs manually instead of using `download_graphs.py`:

### Recommended Graphs

| Symbol | Graph | Size | Format | Download Link |
|--------|-------|------|--------|---------------|
| TWTR | Twitter | 31.4GB | .mtx | [GAP-twitter](https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-twitter.tar.gz) |
| RD | Road Network | 628MB | .mtx | [GAP-road](https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-road.tar.gz) |
| SLJ1 | LiveJournal | 1GB | .mtx | [soc-LiveJournal1](https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-LiveJournal1.tar.gz) |
| CPAT | Patents | 262MB | .mtx | [cit-Patents](https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-Patents.tar.gz) |
| HWOD | Hollywood | 600MB | .mtx | [hollywood-2009](https://suitesparse-collection-website.herokuapp.com/MM/LAW/hollywood-2009.tar.gz) |
| UK02 | UK Web 2002 | 4GB | .mtx | [uk-2002](https://suitesparse-collection-website.herokuapp.com/MM/LAW/uk-2002.tar.gz) |

### Directory Structure

After downloading, organize graphs as:
```
graphs/
├── graphs.json          # Auto-generated config (use download_graphs.py)
├── email-Enron/
│   └── graph.mtx
├── cit-Patents/
│   └── graph.mtx
└── ...
```

# GraphBrew Standalone

## Usage

### Example Usage
   * To compile, run, and then clean up the betweenness centrality benchmark:
```bash
make all
make run-bc
make clean
```
### Compiling a single Benchmarks
   * Where `make <benchmark_name>` can be `bc`, `bfs`, `converter`, etc.
```bash
make bc
```
### Compiling the Benchmarks
   * To build all benchmarks:
```bash
make all
```

### Running a single Benchmarks
   * To run a specific benchmark, use:
```bash
make run-<benchmark_name>
```
   * Where `<benchmark_name>` can be `bc`, `bfs`, `converter`, etc.
```bash
make run-bfs
```
### Parameters Makefile
All parameters [(section)](#graphbrew-parameters) can be passed through the Make command via:
   * `RUN_PARAMS='-n1 -o11'`, for controlling aspects of the algorithm and reordering.
   * `GRAPH_BENCH ='-f ./test/graphs/4.el'`,`GRAPH_BENCH ='-g 4'`, for controlling the graph path, or kron/random generation.
### Parameters Binary
All parameters [(section)](#graphbrew-parameters) can be passed through the binary command via:
   * `./bench/bin/<benchmark_name> -f ./test/graphs/4.el -n1 -o11`
   * `./bench/bin/<benchmark_name> -g 4 -n1 -o11`

### Relabeling the graph
   * `converter` is used to convert graphs and apply new labeling to them.
   * Please check converter parameters and pass them to `RUN_PARAMS='-p ./graph_8.mtx -o 8'`.
```bash
make run-converter GRAPH_BENCH='-f ./graph.<mtx|el|sg>' RUN_PARAMS='-p ./graph_8.mtx -o 8' 
<OR>
./bench/bin/converter -f ./graph.<mtx|el|sg> -p ./graph_8.mtx -o 8
```

### Debugging
   * To run a benchmark with gdb:
```bash
make run-<benchmark_name>-gdb
```
   * To run a benchmark with memory checks (using valgrind):
```bash
make run-<benchmark_name>-mem
```

### Clean up
   * To clean up all compiled files:
```bash
make clean
```
   * To clean up all compiled including results (backed up automatically in `bench/backup`) files:
```bash
make clean-all
```

### Help
   * To display help for a specific benchmark or for general usage:
```bash
make help-<benchmark_name>
make help
```

## Generating Reordered Graphs

### Overview

Use the `make run-converter` command to generate reordered graphs from input graph files. The converter supports various output formats, including serialized graphs, edge lists, Matrix Market exchange format, Ligra adjacency graph format, and reordered labels.

### Command-Line Options

The `CLConvert` class provides several command-line options for generating different output formats. Here is a summary of the options:

- `-b file`: Output serialized graph to file (`.sg`).
- `-e file`: Output edge list to file (`.el`).
- `-p file`: Output Matrix Market exchange format to file (`.mtx`).
- `-y file`: Output in Ligra adjacency graph format to file (`.ligra`).
- `-w file`: Make output weighted (`.wel` | `.wsg`| `.wligra`).
- `-x file`: Output new reordered labels to file list (`.so`).
- `-q file`: Output new reordered labels to file serialized (`.lo`).
- `-o order`: Apply reordering strategy, optionally with a parameter (e.g., `-o 3`, `-o 2`, `-o 14:mapping.label`).

### Example Usage

#### Step 1: Prepare the Input Graph Files

Make sure you have the input graph files (`graph_1.<mtx|el|sg>`) while specifying their paths correctly.

#### Step 2: Run the Converter

Use the `make run-converter` command with the appropriate `GRAPH_BENCH` and `RUN_PARAMS` values to generate the reordered graphs. Here is an example command:

```bash
make run-converter GRAPH_BENCH='-f ./graph.<mtx|el|sg>' RUN_PARAMS='-p ./graph_8.mtx -o 8' 
```

#### Step 3: Specify Output Formats

You can specify multiple output formats by combining the command-line options. Here is an example that generates multiple output formats:

```bash
make run-converter GRAPH_BENCH='-f ./graph.<mtx|el|sg>' RUN_PARAMS='-b graph.sg -e graph.el -p graph.mtx -y graph.ligra -x labels.so -q labels.lo'
```

#### Step 4: Apply Reordering Strategy

To apply a reordering strategy on the newly generated graph, use the `-o` option. For example:

```bash
make run-converter GRAPH_BENCH='-f ./graph.<mtx|el|sg>' RUN_PARAMS='-p ./graph_3_2_14.mtx -o 3 -o 2 -o 14:mapping.<lo|so>'
```

### Combining Multiple Output Formats and Reordering

You can generate multiple output formats and apply reordering in a single command. Here is an example:

```bash
make run-converter GRAPH_BENCH='-f ./graph.<mtx|el|sg>' RUN_PARAMS='-b graph_3.sg -e graph_3.el -p graph_3.mtx -y graph_3.ligra -x labels_3.so -q labels_3.lo -o 3'
```


## Graph Loading

All of the binaries use the same command-line options for loading graphs:
+ `-g 20` generates a Kronecker graph with 2^20 vertices (Graph500 specifications)
+ `-u 20` generates a uniform random graph with 2^20 vertices (degree 16)
+ `-f graph.el` loads graph from file graph.el
+ `-sf graph.el` symmetrizes graph loaded from file graph.el

The graph loading infrastructure understands the following formats:
+ `.el` plain-text edge-list with an edge per line as _node1_ _node2_
+ `.wel` plain-text weighted edge-list with an edge per line as _node1_ _node2_ _weight_
+ `.gr` [9th DIMACS Implementation Challenge](http://www.dis.uniroma1.it/challenge9/download.shtml) format
+ `.graph` Metis format (used in [10th DIMACS Implementation Challenge](http://www.cc.gatech.edu/dimacs10/index.shtml))
+ `.mtx` [Matrix Market](http://math.nist.gov/MatrixMarket/formats.html) format
+ `.sg` serialized pre-built graph (use `converter` to make)
+ `.wsg` weighted serialized pre-built graph (use `converter` to make)

The GraphBrew loading infrastructure understands the following formats for reordering labels:
+ `-o 14:mapping.lo` loads new reodering labels from file `mapping.<lo|so>` is also supported
+ `.so` reordered serialized labels list (.so) (use `converter` to make), _node_id_ per line as _node_label_ 
+ `.lo` reordered plain-text labels list (.lo) (use `converter` to make), _node_id_ per line as _node_label_ 

## GraphBrew Parameters

All parameters can be passed through the make command via:
   * Reorder the graph, orders can be layered.
   * Segment the graph for scalability, requires modifying the algorithm to iterate through segments.
   * `RUN_PARAMS='-n1 -o11'`, for controlling aspects of the algorithm and reordering.
   * `GRAPH_BENCH ='-f ./test/graphs/4.el'`,`GRAPH_BENCH ='-g 4'`, for controlling the graph path, or kron/random generation.

### GAP Parameters (PageRank example)
```bash
make pr
--------------------------------------------------------------------------------
pagerank
 -h           : print this help message                                         
 -f <file>    : load graph from file                                            
 -s           : symmetrize input edge list                               [false]
 -g <scale>   : generate 2^scale kronecker graph                                
 -u <scale>   : generate 2^scale uniform-random graph                           
 -k <degree>  : average degree for synthetic graph                          [16]
 -m           : reduces memory usage during graph building               [false]
 -o <order>   : apply reordering strategy, optionally with a parameter 
               [example]-o 3 -o 2 -r 14:mapping.<lo|so>               [optional]
 -z <indegree>: use indegree for ordering [Degree Based Orderings]       [false]
 -j <segments>: number of segments for the graph 
               [type:n:m] <0:GRAPHIT/Cagra> <1:TRUST>                    [0:1:1]
 -a           : output analysis of last run                              [false]
 -n <n>       : perform n trials                                            [16]
 -r <node>    : start from node r                                         [rand]
 -v           : verify the output of each run                            [false]
 -l           : log performance within each trial                        [false]
 -i <i>       : perform at most i iterations                                [20]
 -t <t>       : use tolerance t                                       [0.000100]
--------------------------------------------------------------------------------
```
### Reorder Parameters
```bash
--------------------------------------------------------------------------------
-o <order>   : Apply reordering strategy with optional parameters
               Format: -o <algo> or -o <algo>:<param1>:<param2>:...
               
-j <segments>: Number of segments for the graph 
               [type:n:m] <0:GRAPHIT/Cagra> <1:TRUST>                    [0:1:1]

-z <indegree>: Use indegree for ordering [Degree Based Orderings]        [false]
--------------------------------------------------------------------------------
Reordering Algorithms:
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Basic Algorithms                                                            │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ ORIGINAL       (0):  No reordering applied                                  │
  │ RANDOM         (1):  Apply random reordering                                │
  │ SORT           (2):  Apply sort-based reordering                            │
  │ HUBSORT        (3):  Apply hub-based sorting                                │
  │ HUBCLUSTER     (4):  Apply clustering based on hub scores                   │
  │ DBG            (5):  Apply degree-based grouping                            │
  │ HUBSORTDBG     (6):  Combine hub sorting with degree-based grouping         │
  │ HUBCLUSTERDBG  (7):  Combine hub clustering with degree-based grouping      │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Community-Based Algorithms                                                  │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ RABBITORDER    (8):  Community clustering with incremental aggregation      │
  │ GORDER         (9):  Dynamic programming BFS and windowing ordering         │
  │ CORDER        (10):  Workload balancing via graph reordering                │
  │ RCM           (11):  Reverse Cuthill-McKee algorithm (BFS-based)            │
  │ LeidenOrder   (12):  Leiden community detection with Louvain refinement     │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Advanced Hybrid Algorithms                                                  │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ GraphBrewOrder(13):  Leiden clustering + configurable per-community order   │
  │ MAP           (14):  Load reordering from file (-o 14:mapping.<lo|so>)      │
  │ AdaptiveOrder (15):  ML-based perceptron selector for optimal algorithm     │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Leiden Dendrogram Variants (NEW)                                            │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ LeidenDFS     (16):  Leiden + DFS standard traversal of dendrogram          │
  │ LeidenDFSHub  (17):  Leiden + DFS prioritizing hub communities first        │
  │ LeidenDFSSize (18):  Leiden + DFS prioritizing larger communities first     │
  │ LeidenBFS     (19):  Leiden + BFS level-order traversal of dendrogram       │
  │ LeidenHybrid  (20):  Leiden + Hybrid hub-aware DFS (RECOMMENDED)            │
  └─────────────────────────────────────────────────────────────────────────────┘

Parameter Syntax for Composite Algorithms:
--------------------------------------------------------------------------------
  GraphBrewOrder (13) - Format: -o 13:<frequency>:<intra_algo>:<resolution>
    <frequency>   : Frequency ordering within communities (default: 10)
                    Options: 0-14 (any basic/community algorithm)
    <intra_algo>  : Algorithm for per-community reordering (default: 8)
                    Options: 0-20 (any algorithm including Leiden variants)
    <resolution>  : Leiden resolution parameter (default: 1.0)
    
    Examples:
      -o 13                    # Default: frequency=10, intra=RabbitOrder
      -o 13:10:17              # Use LeidenDFSHub for per-community ordering
      -o 13:10:20:0.5          # Use LeidenHybrid with resolution 0.5
      
  AdaptiveOrder (15) - Automatically selects optimal algorithm using ML
    Uses graph features (modularity, density, degree variance) to predict
    the best algorithm. No parameters needed.
    
    Example:
      -o 15                    # Let perceptron choose the best algorithm
      
  MAP (14) - Format: -o 14:<mapping_file>
    <mapping_file>: Path to label file (.lo or .so format)
    
    Example:
      -o 14:mapping.lo         # Load reordering from mapping.lo file
```
### Converter Parameters (Generate Optimized Graphs)
```bash
make help-converter
--------------------------------------------------------------------------------
converter
 -h          : print this help message                                         
 -f <file>   : load graph from file                                            
 -s          : symmetrize input edge list                                [false]
 -g <scale>  : generate 2^scale kronecker graph                                
 -u <scale>  : generate 2^scale uniform-random graph                           
 -k <degree> : average degree for synthetic graph                           [16]
 -m          : reduces memory usage during graph building                [false]
 -o <order>  : Apply reordering strategy, optionally layer ordering 
               [example]-o 3 -o 2 -o 14:mapping.<lo|so>               [optional]
 -z <indegree>: use indegree for ordering [Degree Based Orderings]       [false]
 -j <segments>: number of segments for the graph 
               [type:n:m] <0:GRAPHIT/Cagra> <1:TRUST>                    [0:1:1]
 --------------------------------------------------------------------------------
 -b <file>   : output serialized graph to file (.sg)                           
 -V <file>   : output edge list to file (.el)
 -e <file>   : output edge list csr structure individually to files(.out_degree/.out_offset..etc)
 -p <file>   : output matrix market exchange format to file (.mtx)
 -y <file>   : output in Ligra adjacency graph format to file (.ligra)                                      
 -w <file>   : make output weighted (.wel|.wsg)                                
 -x <file>   : output new reordered labels to file list (.so)                  
 -q <file>   : output new reordered labels to file serialized (.lo)    
 --------------------------------------------------------------------------------
```

### Makefile Flow
```bash
available Make commands:
  all            - Builds all targets including GAP benchmarks (CPU)
  run-%          - Runs the specified GAP benchmark (bc bfs cc cc_sv pr pr_spmv sssp tc)
  help-%         - Print the specified Help (bc bfs cc cc_sv pr pr_spmv sssp tc)
  clean          - Removes all build artifacts
  help           - Displays this help message
 --------------------------------------------------------------------------------
Example Usage:
  make all - Compile the program.
  make clean - Clean build files.
  ./bench/bin/pr -g 15 -n 1 -o 14:mapping.lo - Execute with MAP reordering using 'mapping.<lo|so> '.

```

## Modifying the Makefile

### Compiler Setup
- **`CC`**: The C compiler to be used, checks for `gcc-9` first, if not found, falls back to `gcc`.
- **`CXX`**: The C++ compiler to be used, checks for `g++-9` first, if not found, falls back to `g++`.

### Directory Structure
- **`BIN_DIR`**: Directory for compiled binaries.
- **`LIB_DIR`**: Library directory.
- **`SRC_DIR`**: Source files directory.
- **`INC_DIR`**: Include directory for header files.
- **`OBJ_DIR`**: Object files directory.
- **`SCRIPT_DIR`**: Scripts used for operations like graph processing.
- **`BENCH_DIR`**: Benchmark directory.
- **`CONFIG_DIR`**: Configuration files for scripts and full expriments in congif.json format.
- **`RES_DIR`**: Directory where results are stored.
- **`BACKUP_DIR`**: Directory for backups of results `make clean-results`. backsup results then cleans them.

### Include Directories
- **`INCLUDE_<LIBRARY>`**: Each variable specifies the path to header files for various libraries or modules.
- **`INCLUDE_BOOST`**: Specifies the directory for Boost library headers.

### Compiler and Linker Flags
- **`CXXFLAGS`**: Compiler flags for C++ files, combining flags for different libraries and conditions.
- **`LDLIBS`**: Linker flags specifying libraries to link against.
- **`CXXFLAGS_<LIBRARY>`**: Specific compiler flags for various libraries/modules.
- **`LDLIBS_<LIBRARY>`**: Specific linker flags for various libraries/modules.

### Runtime and Execution
- **`PARALLEL`**: Number of parallel threads.
- **`FLUSH_CACHE`**: Whether or not to flush cache before running benchmarks.
- **`GRAPH_BENCH`**: Command line arguments for specifying graph benchmarks.
- **`RUN_PARAMS`**: General command line parameters for running benchmarks.

## Makefile Targets

### Primary Targets
- **`all`**: Compiles all benchmarks.
- **`clean`**: Removes binaries and intermediate files.
- **`clean-all`**: Removes binaries, results, and intermediate files.
- **`clean-results`**: Backs up and then cleans the results directory.
- **`exp-%`**: Runs a specific experiment by replacing `%` with the experiment.json name. E.g., `test.json`.
- **`run-%`**: Runs a specific benchmark by replacing `%` with the benchmark name. E.g., `run-bfs`.
- **`run-%-gdb`**: Runs a specific benchmark under GDB.
- **`run-%-mem`**: Runs a specific benchmark under Valgrind for memory leak checks.
- **`run-all`**: Runs all benchmarks.
- **`graph-%`**: Downloads necessary graphs for a specific benchmark at `CONFIG_DIR`.
- **`help`**: Displays help for all benchmarks.

### Compilation Rules
- **`$(BIN_DIR)/%`**: Compiles a `.cc` source file into a binary, taking dependencies into account.

### Directory Setup
- **`$(BIN_DIR)`**: Ensures the binary directory and required sub directories exist.

### Cleanup
- **`clean`**: Removes binaries and intermediate files.
- **`clean-all`**: Removes binaries, results, and intermediate files.

### Help
- **`help`**: Provides a generic help message about available commands.
- **`help-%`**: Provides specific help for each benchmark command, detailing reordering algorithms and usage examples.

## Project Structure
- `bench/bin`: Executable is placed here.
- `bench/lib`: Library files can be stored here (not used by default).
- `bench/src`: Source code files (*.cc) for the benchmarks.
- `bench/obj`    : Object files are stored here (directory creation is handled but not used by default).
- `bench/include`: Header files for the benchmarks and various include files for libraries such as GAPBS, RABBIT, etc.

## Project Experiments
- `bench/results`: experiment results from running `exp-%`.
- `bench/backups`: experiment backup results from running `clean-all` or `clean-results`.

## Installing Prerequisites (General)

* These tools are available on most Unix-like operating systems and can be installed via your package manager. For example, on Ubuntu, you can install them using:

```bash
sudo apt-get update
sudo apt-get install g++ make libomp-dev
```

## Installing Prerequisites (RabbitOrder)
* Go to Makefile <line:8> make sure `RABBIT_ENABLE = 1`
```bash
<OR>
make RABBIT_ENABLE=1
```

* These made optional if you don't need Rabbit Order or running on machines where you can't install these libraries
```bash
sudo apt-get install libgoogle-perftools-dev
sudo apt-get install python3 python3-pip python3-venv
```
### Installing Boost 1.58.0

1. First, navigate to your project directory


   * Download the desired Boost version `boost_1_58_0`:
```bash
cd ~
wget http://downloads.sourceforge.net/project/boost/boost/1.58.0/boost_1_58_0.tar.gz
tar -zxvf boost_1_58_0.tar.gz
cd boost_1_58_0
```
   * Determine the number of CPU cores available to optimize the compilation process:
```bash
cpuCores=$(cat /proc/cpuinfo | grep "cpu cores" | uniq | awk '{print $NF}')
echo "Available CPU cores: $cpuCores"
```
   * Initialize the Boost installation script:
```bash
./bootstrap.sh --prefix=/opt/boost_1_58_0 --with-python=python2.7 
```
   * Compile and install Boost using all available cores to speed up the process:
```bash
sudo ./b2 --with=all -j $cpuCores install
```

3. **Verify the Installation**

   
   * After installation, verify that Boost has been installed correctly by checking the installed version:
```bash
cat /opt/boost_1_58_0/include/boost/version.hpp | grep "BOOST_LIB_VERSION"
```
   * The output should display the version of Boost you installed, like so:
```bash
//  BOOST_LIB_VERSION must be defined to be the same as BOOST_VERSION
#define BOOST_LIB_VERSION "1_58"
```

How to Cite
-----------

Please cite the following papers if you find this repository useful.
+ S. Beamer, K. Asanović, and D. Patterson, “The GAP Benchmark Suite,” arXiv:1508.03619 [cs], May 2017.
+ J. Arai, H. Shiokawa, T. Yamamuro, M. Onizuka, and S. Iwamura.Rabbit Order: Just-in-time Parallel Reordering for Fast Graph Analysis.
+ P. Faldu, J. Diamond, and B. Grot, “A Closer Look at Lightweight Graph Reordering,” arXiv:2001.08448 [cs], Jan. 2020.
+ V. Balaji, N. Crago, A. Jaleel, and B. Lucia, “P-OPT: Practical Optimal Cache Replacement for Graph Analytics,” in 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA), Feb. 2021, pp. 668–681. doi: 10.1109/HPCA51647.2021.00062.
+ Y. Zhang, V. Kiriansky, C. Mendis, S. Amarasinghe, and M. Zaharia, “Making caches work for graph analytics,” in 2017 IEEE International Conference on Big Data (Big Data), Dec. 2017, pp. 293–302. doi: 10.1109/BigData.2017.8257937.
+ Y. Zhang, M. Yang, R. Baghdadi, S. Kamil, J. Shun, and S. Amarasinghe, “GraphIt: a high-performance graph DSL,” Proc. ACM Program. Lang., vol. 2, no. OOPSLA, p. 121:1-121:30, Oct. 2018, doi: 10.1145/3276491.
+ H. Wei, J. X. Yu, C. Lu, and X. Lin, “Speedup Graph Processing by Graph Ordering,” New York, NY, USA, Jun. 2016, pp. 1813–1828. doi: 10.1145/2882903.2915220.
+ Y. Chen and Y.-C. Chung, “Workload Balancing via Graph Reordering on Multicore Systems,” IEEE Transactions on Parallel and Distributed Systems, 2021.
+ A. George and J. W. H. Liu, Computer Solution of Large Sparse Positive Definite Systems. Prentice-Hall, 1981
+ S. Sahu, “GVE-Leiden: Fast Leiden Algorithm for Community Detection in Shared Memory Setting.” arXiv, Mar. 28, 2024. doi: 10.48550/arXiv.2312.13936.
+ V. A. Traag, L. Waltman, and N. J. van Eck, “From Louvain to Leiden: guaranteeing well-connected communities,” Sci Rep, vol. 9, no. 1, p. 5233, Mar. 2019, doi: 10.1038/s41598-019-41695-z.
