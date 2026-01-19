# GraphBrew Wiki

Welcome to the **GraphBrew** wiki! This comprehensive guide will help you understand, use, and extend the GraphBrew framework for graph reordering and benchmark analysis.

## 🍺 What is GraphBrew?

GraphBrew is a high-performance graph reordering framework that combines **community detection** with **cache-aware vertex reordering** to dramatically improve graph algorithm performance. It implements over 20 different reordering algorithms and provides tools to automatically select the best one for your specific graph.

### Key Features

- **21 Reordering Algorithms**: From simple sorting to advanced ML-based selection
- **Leiden Community Detection**: State-of-the-art community detection for graph partitioning
- **AdaptiveOrder**: ML-powered perceptron that automatically selects the best algorithm
- **Comprehensive Benchmarks**: PageRank, BFS, Connected Components, Triangle Counting, and more
- **Python Analysis Tools**: Correlation analysis, benchmark automation, and weight training

## 📚 Wiki Contents

### Getting Started
- [[Installation]] - System requirements and build instructions
- [[Quick-Start]] - Run your first benchmark in 5 minutes
- [[Supported-Graph-Formats]] - EL, MTX, GRAPH, and other formats

### Understanding the Algorithms
- [[Reordering-Algorithms]] - Complete guide to all 21 algorithms
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

## 🔗 Quick Links

- [GitHub Repository](https://github.com/atmughrabi/GraphBrew)
- [Paper/Citation](#) (if applicable)
- [Issue Tracker](https://github.com/atmughrabi/GraphBrew/issues)

---

*Last updated: January 2026*
