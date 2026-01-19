# Quick Start Guide

Get up and running with GraphBrew in 5 minutes!

## Prerequisites

- Linux or macOS
- GCC 7+ with C++17 support
- Make
- Python 3.6+ (for analysis scripts)

---

## 1. Clone and Build (2 minutes)

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
# Generate performance comparison
cd scripts
python3 graph_brew.py --help
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

[← Back to Home](Home) | [Installation →](Installation)
