# Quick Start Guide

Get up and running with GraphBrew in 5 minutes!

## Prerequisites

- Linux or macOS
- GCC 7+ with C++17 support
- Make
- Python 3.8+ (optional, for analysis scripts - no pip install needed)

---

## 🚀 One-Command Pipeline (Recommended)

Run the full pipeline — download, build, benchmark, and ML evaluation — with a single flag:

```bash
# Clone and run - that's it!
git clone https://github.com/UVA-LavaLab/GraphBrew.git
cd GraphBrew

# Download 150 graphs per size, run full pipeline + ML evaluation
python3 scripts/graphbrew_experiment.py --target-graphs 150
```

`--target-graphs N` auto-enables `--full`, `--catalog-size N`, `--auto`, and `--all-variants`. It will:
- ✅ Download N graphs per size category from SuiteSparse
- ✅ Auto-detect RAM and skip graphs that won't fit
- ✅ Auto-detect disk space and limit downloads accordingly
- ✅ Build all binaries (standard + cache simulation)
- ✅ Pre-generate reordered `.sg` per algorithm (MAP mode — no runtime reorder overhead)
- ✅ Expand Leiden algorithms into all variants (dfs, bfs, hubsort, etc.)
- ✅ Run benchmarks (7 kernels × 18 algorithms × 2 trials)
- ✅ Run LOGO cross-validation and report ML model accuracy
- ✅ Save evaluation summary to `results/data/evaluation_summary.json`

Use `--dry-run` to preview what would happen without executing:

```bash
python3 scripts/graphbrew_experiment.py --target-graphs 150 --dry-run
```

See [[Command-Line-Reference]] for all parameters and [[Benchmark-Suite]] for size categories.

### Examples by Scale

```bash
# Quick test (~10 min): small graphs, skip tiny ones
python3 scripts/graphbrew_experiment.py --target-graphs 30 --size small

# Standard training (~1 hour): medium graphs
python3 scripts/graphbrew_experiment.py --target-graphs 80 --size medium

# Full training (~4-8 hours): all graphs, 150 per size
python3 scripts/graphbrew_experiment.py --target-graphs 150

# Skip ML evaluation at end
python3 scripts/graphbrew_experiment.py --target-graphs 150 --skip-eval
```

### Equivalent Explicit Flags

`--target-graphs N` is shorthand for:

```bash
python3 scripts/graphbrew_experiment.py --full --catalog-size N \
    --auto --all-variants --size all
```

### More Pipeline Examples

```bash
# Just download graphs (no experiments)
python3 scripts/graphbrew_experiment.py --download-only --size medium

# Use pre-generated label maps for faster, consistent reordering
python3 scripts/graphbrew_experiment.py --precompute --phase benchmark

# Skip download phase (use existing graphs)
python3 scripts/graphbrew_experiment.py --full --size large --skip-download

# Clean start (remove all generated data)
python3 scripts/graphbrew_experiment.py --clean-all
python3 scripts/graphbrew_experiment.py --target-graphs 50
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
# Should see: bc bfs cc cc_sv converter pr pr_spmv sssp tc tc_p
```

---

## 2. Get a Test Graph (30 seconds)

```bash
# Use the included test graph (12 edges, 6 nodes)
cat scripts/test/graphs/tiny/tiny.el
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
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -n 3
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
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -o 7 -n 3
```

---

## 4. Try Different Algorithms (1 minute)

### Quick Comparison

```bash
# No reordering (baseline)
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -o 0 -n 3

# Hub-based
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -o 7 -n 3

# Community-based (GraphBrewOrder)
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -o 12 -n 3

# ML-powered selection
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -o 14 -n 3
```

### Algorithm Quick Reference

| ID | Name | Best For |
|----|------|----------|
| 0 | ORIGINAL | Baseline |
| 7 | HUBCLUSTERDBG | General purpose |
| 12 | GraphBrewOrder | Modular graphs |
| 14 | AdaptiveOrder | Auto-selection |
Best algorithm: GraphBrewOrder (`-o 12`, best quality). See [[Reordering-Algorithms]] for all 17 algorithm IDs (13 reorderers + 2 baselines + 2 meta) and variants.

---

## 5. Run All Benchmarks (30 seconds)

```bash
# Run the 7 experiment benchmarks (TC excluded by default in the pipeline)
for bench in pr pr_spmv bfs cc cc_sv sssp bc; do
    echo "=== $bench ==="
    ./bench/bin/$bench -f scripts/test/graphs/tiny/tiny.el -s -o 7 -n 1
done

# Or include TC explicitly:
# ./bench/bin/tc -f scripts/test/graphs/tiny/tiny.el -s -n 1
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
| `-o` | Ordering algorithm (0-16) | `-o 7` |
| `-n` | Number of trials | `-n 5` |
| `-r` | Root vertex (for BFS/SSSP) | `-r 0` |

---

## What's Next?

1. **[[Installation]]** - Detailed installation guide
2. **[[Reordering-Algorithms]]** - Learn about all 17 algorithms
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

## Training Quick Reference

```bash
# Quick test (~10 min)
python3 scripts/graphbrew_experiment.py --train --size small --max-graphs 3 --trials 1

# Standard training (~1 hour)
python3 scripts/graphbrew_experiment.py --train --all-variants --size medium --auto --trials 5

# Check results
ls -la results/data/adaptive_models.json
```

**Interpreting output:** RANDOM is baseline (1.00x). Higher bias = better. Bias < 0.5 = slower than random. See [[Perceptron-Weights]] for details.

---

[← Back to Home](Home) | [Installation →](Installation)
