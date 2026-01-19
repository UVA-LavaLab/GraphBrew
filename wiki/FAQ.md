# Frequently Asked Questions (FAQ)

Common questions and answers about GraphBrew.

---

## General Questions

### What is GraphBrew?

GraphBrew is a graph processing benchmark framework that combines:
- **21 vertex reordering algorithms** for cache optimization
- **6 graph algorithm benchmarks** (PageRank, BFS, CC, SSSP, BC, TC)
- **ML-powered algorithm selection** via AdaptiveOrder
- **Leiden community detection** integration

### Who should use GraphBrew?

- **Researchers** studying graph algorithms and cache optimization
- **Engineers** optimizing graph processing pipelines
- **Students** learning about graph algorithms
- **Data scientists** working with network data

### What makes GraphBrew different?

1. **Comprehensive**: 21 reordering algorithms in one framework
2. **ML-powered**: AdaptiveOrder learns which algorithm works best
3. **Modern**: Leiden community detection integration
4. **Practical**: Based on GAP Benchmark Suite standards

---

## Installation Questions

### What are the requirements?

- Linux or macOS
- GCC 7+ with C++17 support
- Make
- Python 3.6+ (optional, for scripts)
- At least 4GB RAM (more for large graphs)

### How do I install on Ubuntu?

```bash
sudo apt-get update
sudo apt-get install build-essential git
git clone https://github.com/UVA-LavaLab/GraphBrew.git
cd GraphBrew
make all
```

### How do I install on macOS?

```bash
xcode-select --install
brew install gcc
git clone https://github.com/UVA-LavaLab/GraphBrew.git
cd GraphBrew
make all CXX=g++-13
```

### The build fails. What should I check?

1. GCC version: `g++ --version` (need 7+)
2. C++17 support: `g++ -std=c++17 --version`
3. OpenMP: `echo '#include <omp.h>' | g++ -fopenmp -x c++ - -c`

See [[Installation]] for detailed troubleshooting.

---

## Usage Questions

### How do I run a simple benchmark?

```bash
./bench/bin/pr -f graph.el -s -n 3
```

### What does each option mean?

| Option | Meaning |
|--------|---------|
| `-f graph.el` | Input file |
| `-s` | Make undirected (symmetrize) |
| `-n 3` | Run 3 trials |
| `-o 7` | Use algorithm 7 (HUBCLUSTERDBG) |

### Which reordering algorithm should I use?

| Situation | Recommendation |
|-----------|----------------|
| Don't know | `-o 15` (AdaptiveOrder) |
| Social network | `-o 12` (LeidenOrder) |
| General purpose | `-o 7` (HUBCLUSTERDBG) |
| Large graph | `-o 20` (LeidenHybrid) |
| Baseline | `-o 0` (no reordering) |

### How do I know which algorithm is best for my graph?

Run multiple algorithms and compare:

```bash
for algo in 0 7 12 15 20; do
    echo "=== Algorithm $algo ==="
    ./bench/bin/pr -f graph.el -s -o $algo -n 3
done
```

Or use AdaptiveOrder (`-o 15`) to auto-select.

### What graph formats are supported?

| Format | Extension | Example |
|--------|-----------|---------|
| Edge list | `.el` | `0 1` |
| Weighted | `.wel` | `0 1 2.5` |
| Matrix Market | `.mtx` | Standard MTX |
| DIMACS | `.gr` | Road networks |

See [[Supported-Graph-Formats]] for details.

---

## Performance Questions

### Why is reordering slow?

Reordering is preprocessing that pays off over multiple algorithm runs:

| Operation | Time |
|-----------|------|
| Load graph | 1x |
| Reorder | 0.5-2x |
| Benchmark | 0.7-0.9x (faster!) |

For repeated analyses, reorder once, save, reuse.

### How much speedup should I expect?

Typical improvements:

| Graph Type | PageRank | BFS | TC |
|------------|----------|-----|-----|
| Social | 1.2-1.5x | 1.1-1.3x | 1.5-3x |
| Web | 1.3-1.8x | 1.2-1.4x | 2-4x |
| Road | 1.0-1.1x | 1.0-1.1x | 1.1-1.3x |

### Why is my graph loading slowly?

1. **Large file**: Use binary format
   ```bash
   ./bench/bin/converter -f graph.el -s -o graph.sg
   ./bench/bin/pr -f graph.sg -n 3
   ```

2. **Text parsing**: MTX/EL requires parsing; binary is instant

3. **Memory**: Ensure sufficient RAM

### How can I make benchmarks faster?

1. **Use binary graphs** for repeated runs
2. **Tune thread count**: `export OMP_NUM_THREADS=8`
3. **Use NUMA binding**: `numactl --cpunodebind=0`
4. **Reduce trials** during development: `-n 1`

---

## Algorithm Questions

### What is Leiden community detection?

Leiden is a community detection algorithm that finds densely connected groups of vertices. It improves on the popular Louvain algorithm with:
- Guaranteed connected communities
- Better quality partitions
- Faster convergence

GraphBrew uses Leiden to guide reordering decisions.

### How does AdaptiveOrder work?

1. Detects communities using Leiden
2. Computes features for each community
3. Uses ML perceptron to select best algorithm per community
4. Applies different reorderings to different parts of the graph

See [[AdaptiveOrder-ML]] for details.

### What's the difference between LeidenOrder and LeidenHybrid?

| Algorithm | Approach |
|-----------|----------|
| LeidenOrder (12) | Basic Leiden → contiguous community ordering |
| LeidenDFS (16) | Leiden + DFS traversal within communities |
| LeidenHybrid (20) | Leiden + multiple strategies + best selection |

LeidenHybrid is most sophisticated but has more overhead.

### When should I use DBG vs HUBCLUSTER?

| Algorithm | Best For |
|-----------|----------|
| DBG | Power-law graphs with clear hot/cold separation |
| HUBCLUSTER | Graphs where hubs connect to each other |
| HUBCLUSTERDBG | Combines both - good general choice |

---

## Troubleshooting

### "Segmentation fault" when loading graph

1. **Check file exists**: `ls -la graph.el`
2. **Check format**: `head -5 graph.el`
3. **Check indexing**: Vertices should start at 0
4. **Check memory**: `free -h`

### "Invalid graph format" error

```bash
# Check for issues
head -5 graph.el        # Should be: node1 node2
file graph.el           # Should be: ASCII text
dos2unix graph.el       # Fix Windows line endings
```

### Results vary between runs

This is normal! Use more trials (`-n 10`) and report averages.

Reduce variance:
- Disable CPU frequency scaling
- Run on dedicated machine
- Use `numactl` for memory binding

### Algorithm X isn't giving speedup

Not all algorithms help all graphs:
- Road networks often don't benefit much
- Small graphs have minimal cache effects
- Already well-ordered graphs may not improve

Try different algorithms or use AdaptiveOrder.

### Python scripts not working

```bash
# Install dependencies
pip install -r scripts/requirements.txt

# Check Python version
python3 --version  # Need 3.6+

# Check path
cd scripts && python3 graph_brew.py --help
```

---

## Development Questions

### How do I add a new reordering algorithm?

See [[Adding-New-Algorithms]] for a complete guide:
1. Add enum value in `builder.h`
2. Implement reorder function
3. Add switch case
4. (Optional) Add perceptron weights

### How do I add a new benchmark?

See [[Adding-New-Benchmarks]] for a complete guide:
1. Create `bench/src/my_algo.cc`
2. Implement algorithm
3. Add to Makefile
4. Test

### How do I contribute?

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

See [CONTRIBUTING.md](https://github.com/UVA-LavaLab/GraphBrew/blob/main/CONTRIBUTING.md) for guidelines.

### What's the code license?

GraphBrew is released under the MIT License. See [LICENSE](https://github.com/UVA-LavaLab/GraphBrew/blob/main/LICENSE).

---

## Data Questions

### Where can I download graphs?

| Source | URL | Formats |
|--------|-----|---------|
| SNAP | snap.stanford.edu/data | Edge list |
| SuiteSparse | sparse.tamu.edu | MTX |
| Network Repository | networkrepository.com | Various |
| KONECT | konect.cc | Various |

### How do I convert my graph format?

```bash
# CSV to edge list
cat graph.csv | tr ',' ' ' | tail -n +2 > graph.el

# MTX to edge list (1-indexed to 0-indexed)
grep -v "^%" graph.mtx | tail -n +2 | awk '{print $1-1, $2-1}' > graph.el
```

### What's the maximum graph size?

Limited by RAM:
- ~16 bytes per edge
- 1B edges ≈ 16GB RAM
- Larger graphs need out-of-core processing (not supported)

---

## Still Have Questions?

- Check [[Home]] for documentation overview
- Review [[Troubleshooting]] for common issues
- Open a GitHub issue for bugs
- Start a discussion for questions

---

[← Back to Home](Home)
