# GraphBrew Wiki

Welcome to the **GraphBrew** wiki! This comprehensive guide will help you understand, use, and extend the GraphBrew framework for graph reordering and benchmark analysis.

## 🍺 What is GraphBrew?

GraphBrew is a high-performance graph reordering framework that combines **community detection** with **cache-aware vertex reordering** to dramatically improve graph algorithm performance. It implements over 20 different reordering algorithms and provides tools to automatically select the best one for your specific graph.

### Key Features

- **20 Reordering Algorithms**: From simple sorting to advanced ML-based selection (IDs 0-13, 15-20)
- **Leiden Community Detection**: State-of-the-art community detection for graph partitioning
- **AdaptiveOrder**: ML-powered perceptron that automatically selects the best algorithm
- **Comprehensive Benchmarks**: PageRank, BFS, Connected Components, SSSP, BC (5 automated), plus Triangle Counting (manual)
- **Python Analysis Tools**: Correlation analysis, benchmark automation, and weight training
- **Iterative Training**: Feedback loop to optimize adaptive algorithm selection

## 📚 Wiki Contents

### Getting Started
- [[Installation]] - System requirements and build instructions
- [[Quick-Start]] - Run your first benchmark in 5 minutes
- [[Supported-Graph-Formats]] - EL, MTX, GRAPH, and other formats

### Understanding the Algorithms
- [[Reordering-Algorithms]] - Complete guide to all 20 algorithms
- [[Graph-Benchmarks]] - PageRank, BFS, CC, SSSP, BC, TC explained
- [[Community-Detection]] - How Leiden clustering works

### Running Experiments
- [[Running-Benchmarks]] - Command-line usage and options
- [[Benchmark-Suite]] - Automated experiment runner
- [[Correlation-Analysis]] - Finding the best algorithm for your graphs

### Advanced Topics
- [[AdaptiveOrder-ML]] - The perceptron-based algorithm selector
- [[Perceptron-Weights]] - Training and tuning the ML model
- [[GraphBrewOrder]] - Per-community reordering explained
- [[Cache-Simulation]] - Cache performance analysis for algorithms

### Developer Guide
- [[Adding-New-Algorithms]] - Implement your own reordering method
- [[Adding-New-Benchmarks]] - Add new graph algorithms
- [[Code-Architecture]] - Understanding the codebase structure
- [[Python-Scripts]] - Analysis and utility scripts

### Reference
- [[Command-Line-Reference]] - All flags and options
- [[Configuration-Files]] - JSON configs and settings
- [[Troubleshooting]] - Common issues and solutions
- [[FAQ]] - Frequently asked questions

## 🚀 Quick Example

### One-Click Full Pipeline (Recommended)

```bash
# Clone, download graphs, build, and run complete experiment
git clone https://github.com/UVA-LavaLab/GraphBrew.git
cd GraphBrew
python3 scripts/graphbrew_experiment.py --full --download-size SMALL
```

This single command will:
1. Download benchmark graphs from SuiteSparse (87 graphs available)
2. Build binaries automatically
3. Pre-generate label mappings for consistent reordering
4. Run all benchmarks with all 20 algorithms
5. Execute cache simulations (L1/L2/L3 hit rates)
6. Generate perceptron weights for AdaptiveOrder (includes cache + reorder time features)

### Resource Management

```bash
# Auto-detect RAM and disk limits
python3 scripts/graphbrew_experiment.py --full --download-size ALL --auto-memory --auto-disk

# Set explicit limits
python3 scripts/graphbrew_experiment.py --full --download-size ALL --max-memory 32 --max-disk 100
```

### Training Options

```bash
# Fill ALL weight fields (cache impacts, topology features, per-benchmark weights)
# Also generates per-graph-type weights (social, road, web, powerlaw, uniform)
python3 scripts/graphbrew_experiment.py --fill-weights --graphs small --max-graphs 5

# Iterative training to reach target accuracy
python3 scripts/graphbrew_experiment.py --train-adaptive --target-accuracy 85 --graphs small
```

**Graph Type Detection:** AdaptiveOrder automatically detects graph types from computed properties (modularity, degree variance, hub concentration) and loads specialized weights for each type.

### Download Size Options

| Size | Graphs | Total | Use Case |
|------|--------|-------|----------|
| `SMALL` | 16 | ~62 MB | Quick testing |
| `MEDIUM` | 28 | ~1.1 GB | Standard experiments |
| `LARGE` | 37 | ~25 GB | Full evaluation |
| `XLARGE` | 6 | ~63 GB | Massive-scale testing |
| `ALL` | **87** | ~89 GB | Complete benchmark |

### Manual Usage

```bash
# Build GraphBrew
make all

# Run PageRank with LeidenHybrid reordering
./bench/bin/pr -f your_graph.el -s -o 20 -n 3

# Let AdaptiveOrder choose the best algorithm
./bench/bin/pr -f your_graph.el -s -o 15 -n 3
```

## 📊 Performance Overview

GraphBrew typically achieves:
- **1.2-3x speedup** on social networks (high modularity)
- **1.1-1.5x speedup** on web graphs
- **1.0-1.2x speedup** on road networks (low modularity)

The best algorithm depends on your graph's topology!

## � Page Index

| Page | Description |
|------|-------------|
| [[Home]] | This page - wiki overview |
| [[Installation]] | Build requirements and instructions |
| [[Quick-Start]] | 5-minute getting started guide |
| [[Supported-Graph-Formats]] | EL, MTX, GRAPH format specs |
| [[Reordering-Algorithms]] | All 21 algorithms explained |
| [[Graph-Benchmarks]] | PR, BFS, CC, SSSP, BC, TC |
| [[Community-Detection]] | Leiden algorithm details |
| [[Running-Benchmarks]] | Manual benchmark execution |
| [[Benchmark-Suite]] | Automated experiment runner |
| [[Correlation-Analysis]] | Feature-algorithm correlation |
| [[AdaptiveOrder-ML]] | ML-based algorithm selection |
| [[Perceptron-Weights]] | Weight file format & tuning |
| [[GraphBrewOrder]] | Per-community hub ordering |
| [[Cache-Simulation]] | Cache performance analysis |
| [[Adding-New-Algorithms]] | Developer: add algorithms |
| [[Adding-New-Benchmarks]] | Developer: add benchmarks |
| [[Code-Architecture]] | Codebase structure |
| [[Python-Scripts]] | Analysis & utility tools |
| [[Command-Line-Reference]] | All CLI flags |
| [[Configuration-Files]] | JSON config reference |
| [[Troubleshooting]] | Common issues |
| [[FAQ]] | Frequently asked questions |

## 🔗 Quick Links

- [GitHub Repository](https://github.com/UVA-LavaLab/GraphBrew)
- [Issue Tracker](https://github.com/UVA-LavaLab/GraphBrew/issues)
- [Contributing Guide](https://github.com/UVA-LavaLab/GraphBrew/blob/main/CONTRIBUTING.md)

---

*Last updated: January 2026*
