# Quick Start Guide

Get up and running with GraphBrew in 5 minutes!

## Prerequisites

- Linux or macOS
- GCC 7+ with C++17 support
- Make
- Python 3.8+ (optional, for analysis scripts - no pip install needed)

---

## 🚀 One-Command Training (Recommended)

Train perceptron weights with all algorithm variants in a single command:

```bash
# Clone and train - that's it!
git clone https://github.com/UVA-LavaLab/GraphBrew.git
cd GraphBrew

# Full training with ALL graphs, all variants, 5 trials
python3 scripts/graphbrew_experiment.py --train --all-variants \
    --size all --min-edges 100000 --auto --trials 5
```

This will automatically:
- ✅ Download benchmark graphs from SuiteSparse (87 graphs available)
- ✅ Auto-detect RAM and skip graphs that won't fit
- ✅ Auto-detect disk space and limit downloads accordingly
- ✅ Skip small graphs (<100k edges) that introduce training noise
- ✅ Build all binaries (standard + cache simulation)
- ✅ Expand Leiden algorithms into all variants (dfs, bfs, hubsort, etc.)
- ✅ Run 5 trials per benchmark for statistical reliability
- ✅ Train per-graph-type perceptron weights for AdaptiveOrder

### Key Parameters Explained

| Parameter | Description |
|-----------|-------------|
| `--train` | Train perceptron weights (runs complete pipeline) |
| `--all-variants` | Test ALL algorithm variants (Leiden, RabbitOrder) |
| `--size SIZE` | Graph size category: `small`, `medium`, `large`, `xlarge`, `all` |
| `--min-edges N` | Skip graphs with fewer than N edges (reduces noise) |
| `--auto` | Auto-detect RAM and disk space limits |
| `--trials N` | Number of benchmark trials (default: 2, recommended: 5) |
| `--skip-download` | Skip graph download phase (use existing graphs) |

### Algorithm Variant Options

| Parameter | Description |
|-----------|-------------|
| `--all-variants` | Test ALL algorithm variants (Leiden, RabbitOrder) |
| `--csr-variants LIST` | LeidenCSR variants: gve (default), gveopt, gverabbit, dfs, bfs, hubsort, fast, modularity |
| `--dendrogram-variants LIST` | LeidenDendrogram variants: dfs, dfshub, dfssize, bfs, hybrid |
| `--rabbit-variants LIST` | RabbitOrder variants: csr (default), boost (requires libboost-graph-dev) |
| `--resolution FLOAT` | Leiden resolution - higher = more communities (default: 1.0) |
| `--passes INT` | LeidenCSR refinement passes (default: 3) |

### Size Categories

| Size | Graphs | Total Size | Use Case |
|------|--------|------------|----------|
| `small` | 16 | ~62 MB | Quick testing (communication, p2p, social) |
| `medium` | 28 | ~1.1 GB | Standard experiments (web, road, mesh, synthetic) |
| `large` | 37 | ~25 GB | Full evaluation (social, web, collaboration) |
| `xlarge` | 6 | ~63 GB | Massive graphs (twitter7, webbase, Kronecker) |
| `all` | **87** | ~89 GB | Complete benchmark set |

### Training Examples by Scale

```bash
# Quick test (~10 min): small graphs, skip tiny ones
python3 scripts/graphbrew_experiment.py --train --all-variants \
    --size small --min-edges 50000 --auto --trials 3

# Standard training (~1 hour): medium graphs
python3 scripts/graphbrew_experiment.py --train --all-variants \
    --size medium --min-edges 100000 --auto --trials 5

# Full training (~4-8 hours): all graphs
python3 scripts/graphbrew_experiment.py --train --all-variants \
    --size all --min-edges 100000 --auto --trials 5
```

---

## 🎯 Quick Full Pipeline

Run the complete GraphBrew experiment with a single command:

```bash
python3 scripts/graphbrew_experiment.py --full --size small --auto
```

### More Pipeline Examples

```bash
# Download medium graphs and run full pipeline
python3 scripts/graphbrew_experiment.py --full --size medium --auto

# Just download graphs (no experiments)
python3 scripts/graphbrew_experiment.py --download-only --size medium

# Use pre-generated label maps for faster, consistent reordering
python3 scripts/graphbrew_experiment.py --generate-maps --use-maps --phase benchmark

# Skip download phase (use existing graphs)
python3 scripts/graphbrew_experiment.py --full --size large --skip-download

# Clean start (remove all generated data)
python3 scripts/graphbrew_experiment.py --clean-all
python3 scripts/graphbrew_experiment.py --full --size small --auto
```

---

## ⚡ Faster Training Options

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
# Should see: bc bfs cc cc_sv converter pr pr_spmv sssp tc tc_p
```

---

## 2. Get a Test Graph (30 seconds)

```bash
# Use the included test graph (12 edges, 6 nodes)
cat test/graphs/5.el
# 0 2
# 2 5
# 4 0
# 4 1
# 4 2
# 5 1
# ...
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
Read Time:           0.041xx
Build Time:          0.041xx
Graph has 14 nodes and 53 undirected edges for degree: 3 Estimated size: 0 MB
Trial Time:          0.000xx
Trial Time:          0.000xx
Trial Time:          0.004xx
Average Time:        0.001xx
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

# Community-based (GraphBrewOrder)
./bench/bin/pr -f test/graphs/4.el -s -o 12 -n 3

# ML-powered selection
./bench/bin/pr -f test/graphs/4.el -s -o 14 -n 3
```

### Algorithm Quick Reference

| ID | Name | Best For |
|----|------|----------|
| 0 | ORIGINAL | Baseline comparison |
| 7 | HUBCLUSTERDBG | General purpose, good default |
| 8 | RabbitOrder | Community-based (variants: csr/boost, default: csr) |
| 12 | GraphBrewOrder | Per-community reordering |
| 14 | AdaptiveOrder | Auto-selection for unknown graphs |
| 15 | LeidenOrder | Social networks |
| 17 | LeidenCSR | Large complex graphs (variants: gve/gveopt/gverabbit/dfs/bfs/hubsort/fast/modularity, default: gve) |

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
python3 scripts/graphbrew_experiment.py --brute-force --size small

# Train adaptive weights with iterative learning
python3 scripts/graphbrew_experiment.py --train-iterative --size small --target-accuracy 50
```

---

## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `-f` | Input graph file | `-f graph.el` |
| `-s` | Symmetrize (make undirected) | `-s` |
| `-o` | Ordering algorithm (0-17) | `-o 7` |
| `-n` | Number of trials | `-n 5` |
| `-r` | Root vertex (for BFS/SSSP) | `-r 0` |

---

## What's Next?

1. **[[Installation]]** - Detailed installation guide
2. **[[Reordering-Algorithms]]** - Learn about all 18 algorithms
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
python3 scripts/graphbrew_experiment.py --size small --max-graphs 3 --trials 1

# TRAIN WEIGHTS: Complete training pipeline (30-60 min)
python3 scripts/graphbrew_experiment.py --train --size small --max-graphs 5 --trials 2

# FULL: Train on all graphs with validation (1-2 hours)
python3 scripts/graphbrew_experiment.py --train --size medium --max-graphs 50 --trials 2 && \
python3 scripts/graphbrew_experiment.py --brute-force --validation-benchmark pr --trials 2

# CHECK RESULTS: View type weight files
ls -la scripts/weights/active/
cat scripts/weights/active/type_0.json | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'{k}: {v.get(\"bias\",0):.2f}') for k,v in sorted(d.items(), key=lambda x:-x[1].get('bias',0))[:10] if not k.startswith('_')]"
```

### Understanding Output

- **RANDOM is baseline** (1.00x) - all speedups measured against random ordering
- **Higher bias = better** - HUBSORT at 26.0 means 26x faster than random
- **Bias < 0.5** - algorithm is slower than random (not useful)

See [Perceptron-Weights](Perceptron-Weights.md#step-by-step-terminal-training-guide) for detailed guide.

---

[← Back to Home](Home) | [Installation →](Installation)
