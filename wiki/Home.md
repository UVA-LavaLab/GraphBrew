# GraphBrew Wiki

Welcome to the **GraphBrew** wiki! This comprehensive guide will help you understand, use, and extend the GraphBrew framework for graph reordering and benchmark analysis.

## 🍺 What is GraphBrew?

GraphBrew is a high-performance graph reordering framework that combines **community detection** with **cache-aware vertex reordering** to dramatically improve graph algorithm performance. It implements **16 algorithm IDs** (0-15): 2 baselines (ORIGINAL, RANDOM), 12 reordering algorithms, and 2 reserved meta-algorithms (MAP, AdaptiveOrder).

### Key Features

- **16 Algorithm IDs**: From simple sorting to advanced ML-based selection (IDs 0-15; 14 benchmark-eligible, 12 produce reorderings)
- **Leiden Community Detection**: State-of-the-art community detection for graph partitioning
- **AdaptiveOrder**: ML-powered perceptron with 15 linear features, 3 quadratic cross-terms, convergence-aware scoring, OOD guardrail, and LOGO cross-validation
- **Comprehensive Benchmarks**: 8 available (PR, PR_SPMV, BFS, CC, CC_SV, SSSP, BC, TC); experiments default to 7 (TC excluded — combinatorial counting, not cache-sensitive traversal)
- **Random Baseline**: Graphs auto-converted to `.sg` with RANDOM ordering so all measurements are relative to a worst-case baseline
- **Pre-generated Reordered .sg**: Each algorithm's reordered graph is pre-generated as `{graph}_{ALGO}.sg` and loaded at benchmark time with `-o 0`, eliminating runtime reorder overhead
- **Amortization Analysis**: Break-even iterations (N*), end-to-end speedup at N, and minimum efficient workload (MinN@95%)
- **Python Analysis Tools**: Correlation analysis, benchmark automation, and weight training with multi-restart perceptrons, regret-aware optimization, and weight evaluation
- **Iterative Training**: Feedback loop to optimize adaptive algorithm selection

## 📚 Wiki Contents

### Getting Started
- [[Installation]] - System requirements and build instructions
- [[Quick-Start]] - Run your first benchmark in 5 minutes
- [[Supported-Graph-Formats]] - EL, MTX, GRAPH, and other formats

### Understanding the Algorithms
- [[Reordering-Algorithms]] - Complete guide to all 16 algorithms
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
python3 scripts/graphbrew_experiment.py --full --size small
```

This single command will:
1. Download benchmark graphs from SuiteSparse (87 graphs available)
2. Build binaries automatically
3. Convert `.mtx` → `.sg` with RANDOM baseline ordering
4. Pre-generate reordered `.sg` per algorithm (12 algorithms, loaded at benchmark time with no runtime reorder overhead)
5. Run all benchmarks with all 14 eligible algorithms
6. Execute cache simulations (L1/L2/L3 hit rates)
7. Generate perceptron weights for AdaptiveOrder (includes cache + reorder time features)

### Options

```bash
# Auto-detect RAM and disk limits
python3 scripts/graphbrew_experiment.py --full --size all --auto

# Train perceptron weights
python3 scripts/graphbrew_experiment.py --train --size small

# Manual: run PageRank with GraphBrewOrder
./bench/bin/pr -f your_graph.el -s -o 12 -n 3

# Let AdaptiveOrder choose the best algorithm
./bench/bin/pr -f your_graph.el -s -o 14 -n 3
```

See [[Quick-Start]] for detailed examples, [[Benchmark-Suite]] for size categories, and [[Command-Line-Reference]] for all flags.

## 📊 Performance Overview

GraphBrew's performance gains depend on your graph's topology and the benchmark algorithm. Run the full pipeline on your target graphs to measure actual speedups.

The best reordering algorithm depends on your graph's structure — see [[Correlation-Analysis]] for details.

## 📋 Page Index

| Page | Description |
|------|-------------|
| [[Home]] | This page - wiki overview |
| [[Installation]] | Build requirements and instructions |
| [[Quick-Start]] | 5-minute getting started guide |
| [[Supported-Graph-Formats]] | EL, MTX, GRAPH format specs |
| [[Reordering-Algorithms]] | All 16 algorithms explained |
| [[Graph-Benchmarks]] | PR, PR_SPMV, BFS, CC, CC_SV, SSSP, BC, TC |
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

*Last updated: February 2026*
