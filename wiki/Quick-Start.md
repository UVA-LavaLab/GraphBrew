# Quick Start Guide

Get up and running with GraphBrew in 5 minutes!

## Prerequisites

- Linux or macOS
- GCC 7+ with C++17 support
- Make
- Python 3.6+ (for analysis scripts)

---

## 🚀 One-Click Full Pipeline (Easiest)

Run the complete GraphBrew experiment with a single command:

```bash
# Clone and run - that's it!
git clone https://github.com/UVA-LavaLab/GraphBrew.git
cd GraphBrew
python3 scripts/graphbrew_experiment.py --full --download-size SMALL
```

This will automatically:
- ✅ Download benchmark graphs from SuiteSparse (96 graphs available)
- ✅ Build all binaries (standard + cache simulation)
- ✅ Pre-generate label mappings for consistent reordering
- ✅ Run performance benchmarks (PR, BFS, CC, SSSP, BC) with all 20 algorithms
- ✅ Execute cache simulations (L1/L2/L3 hit rates)
- ✅ Detect graph types from computed properties (modularity, degree variance, etc.)
- ✅ Train per-graph-type perceptron weights for AdaptiveOrder

All results saved to `./results/` directory, with per-type weights synced to `./scripts/`.

### Download Size Options

| Size | Graphs | Total Size | Use Case |
|------|--------|------------|----------|
| `SMALL` | 16 | ~62 MB | Quick testing (communication, p2p, social) |
| `MEDIUM` | 34 | ~1.1 GB | Standard experiments (web, road, mesh, synthetic) |
| `LARGE` | 40 | ~27 GB | Full evaluation (social, web, collaboration) |
| `XLARGE` | 6 | ~63 GB | Massive graphs (twitter7, webbase, Kronecker) |
| `ALL` | **96** | ~91 GB | Complete benchmark set |

### Memory Management

GraphBrew automatically detects system RAM and can skip graphs that won't fit:

```bash
# Auto-detect RAM and skip graphs that won't fit
python3 scripts/graphbrew_experiment.py --full --download-size ALL --auto-memory

# Set explicit memory limit (e.g., 32 GB)
python3 scripts/graphbrew_experiment.py --full --download-size ALL --max-memory 32
```

### More Examples

```bash
# Download medium graphs and run
python3 scripts/graphbrew_experiment.py --full --download-size MEDIUM

# Just download graphs (no experiments)
python3 scripts/graphbrew_experiment.py --download-only --download-size MEDIUM

# Use pre-generated label maps for faster, consistent reordering
python3 scripts/graphbrew_experiment.py --generate-maps --use-maps --phase benchmark

# Clean start (remove all generated data)
python3 scripts/graphbrew_experiment.py --clean-all --full --download-size SMALL
```

---

## Manual Setup (Step-by-Step)

### 1. Clone and Build (2 minutes)

```bash
# Clone the repository
git clone https://github.com/UVA-LavaLab/GraphBrew.git
cd GraphBrew

# Build everything
make all

# Verify build
ls bench/bin/
# Should see: pr bfs cc sssp bc tc
```

---

## 2. Get a Test Graph (30 seconds)

```bash
# Use the included test graph
cat test/graphs/4.el
# 0 1
# 0 2
# 1 2
# 2 3
```

Or download a real graph:
```bash
# Download a small social network
wget https://snap.stanford.edu/data/ego-Facebook.txt.gz
gunzip ego-Facebook.txt.gz
mv ego-Facebook.txt facebook.el
```

---

## 3. Run Your First Benchmark (1 minute)

### PageRank with Default (No Reordering)

```bash
./bench/bin/pr -f test/graphs/4.el -s -n 3
```

Output:
```
Loading graph from test/graphs/4.el...
Graph has 4 nodes and 4 edges
PageRank completed in 0.001 seconds
```

### PageRank with Reordering

```bash
# Using HUBCLUSTERDBG (algorithm 7)
./bench/bin/pr -f test/graphs/4.el -s -o 7 -n 3
```

---

## 4. Try Different Algorithms (1 minute)

### Quick Comparison

```bash
# No reordering (baseline)
./bench/bin/pr -f test/graphs/4.el -s -o 0 -n 3

# Hub-based
./bench/bin/pr -f test/graphs/4.el -s -o 7 -n 3

# Community-based (Leiden)
./bench/bin/pr -f test/graphs/4.el -s -o 12 -n 3

# ML-powered selection
./bench/bin/pr -f test/graphs/4.el -s -o 15 -n 3
```

### Algorithm Quick Reference

| ID | Name | Best For |
|----|------|----------|
| 0 | ORIGINAL | Baseline comparison |
| 7 | HUBCLUSTERDBG | General purpose, good default |
| 12 | LeidenOrder | Social networks |
| 15 | AdaptiveOrder | Auto-selection for unknown graphs |
| 20 | LeidenHybrid | Large complex graphs |

---

## 5. Run All Benchmarks (30 seconds)

```bash
# Create a simple test script
for bench in pr bfs cc sssp bc tc; do
    echo "=== $bench ==="
    ./bench/bin/$bench -f test/graphs/4.el -s -o 7 -n 1
done
```

---

## 6. Analyze Results (Optional)

```bash
# View all available options
python3 scripts/graphbrew_experiment.py --help

# Run brute-force validation
python3 scripts/graphbrew_experiment.py --brute-force --graphs small

# Train adaptive weights with iterative learning
python3 scripts/graphbrew_experiment.py --train-adaptive --graphs small --target-accuracy 50
```

---

## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `-f` | Input graph file | `-f graph.el` |
| `-s` | Symmetrize (make undirected) | `-s` |
| `-o` | Ordering algorithm (0-20) | `-o 7` |
| `-n` | Number of trials | `-n 5` |
| `-r` | Root vertex (for BFS/SSSP) | `-r 0` |

---

## What's Next?

1. **[[Installation]]** - Detailed installation guide
2. **[[Reordering-Algorithms]]** - Learn about all 21 algorithms
3. **[[Graph-Benchmarks]]** - Understanding the benchmarks
4. **[[Running-Benchmarks]]** - Advanced usage
5. **[[AdaptiveOrder-ML]]** - ML-powered algorithm selection

---

## Quick Troubleshooting

### Build Fails

```bash
# Check compiler version
g++ --version  # Need 7+

# Try with explicit C++17
make CXXFLAGS="-std=c++17"
```

### Graph Won't Load

```bash
# Check format (should be: node1 node2)
head -5 your_graph.el

# Try symmetrizing
./bench/bin/pr -f your_graph.el -s
```

### Permission Denied

```bash
chmod +x bench/bin/*
```

---

## Training Your Own Weights (Terminal Quick Reference)

### One-Line Commands

```bash
# QUICK: Test on 3 small graphs (5-10 min)
python3 scripts/graphbrew_experiment.py --graphs-dir ./results/graphs --graphs small --max-graphs 3 --trials 1

# FILL ALL WEIGHTS: Populate cache impacts, topology features, benchmark weights (30-60 min)
python3 scripts/graphbrew_experiment.py --fill-weights --graphs-dir ./results/graphs --graphs small --max-graphs 5 --trials 2

# FULL: Train on all graphs with validation (1-2 hours)
python3 scripts/graphbrew_experiment.py --graphs-dir ./results/graphs --max-graphs 50 --trials 2 && \
python3 scripts/graphbrew_experiment.py --graphs-dir ./results/graphs --brute-force --bf-benchmark pr --trials 2

# CHECK RESULTS: View new algorithm biases
cat results/perceptron_weights.json | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'{k}: {v.get(\"bias\",0):.2f}') for k,v in sorted(d.items(), key=lambda x:-x[1].get('bias',0))[:10] if not k.startswith('_')]"
```

### Understanding Output

- **RANDOM is baseline** (1.00x) - all speedups measured against random ordering
- **Higher bias = better** - HUBSORT at 26.0 means 26x faster than random
- **Bias < 0.5** - algorithm is slower than random (not useful)

See [Perceptron-Weights](Perceptron-Weights.md#step-by-step-terminal-training-guide) for detailed guide.

---

[← Back to Home](Home) | [Installation →](Installation)
