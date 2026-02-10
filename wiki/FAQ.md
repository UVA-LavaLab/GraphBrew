# Frequently Asked Questions (FAQ)

Common questions and answers about GraphBrew.

---

## General Questions

### What is GraphBrew?

GraphBrew is a graph processing benchmark framework that combines:
- **17 vertex reordering algorithms** (IDs 0-16) for cache optimization
- **6 benchmarks** (PageRank, BFS, CC, SSSP, BC, TC) — 5 run by default, TC binary available separately
- **ML-powered algorithm selection** via AdaptiveOrder
- **Leiden community detection** integration

### Who should use GraphBrew?

- **Researchers** studying graph algorithms and cache optimization
- **Engineers** optimizing graph processing pipelines
- **Students** learning about graph algorithms
- **Data scientists** working with network data

### What makes GraphBrew different?

1. **Comprehensive**: 17 reordering algorithms in one framework
2. **ML-powered**: AdaptiveOrder learns which algorithm works best
3. **Modern**: Leiden community detection integration
4. **Practical**: Based on GAP Benchmark Suite standards

---

## Installation Questions

### What are the requirements?

- Linux or macOS
- GCC 7+ with C++17 support
- Make
- Python 3.8+ (optional, for scripts - no pip dependencies required)
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
| Don't know | `-o 14` (AdaptiveOrder) |
| Social network | `-o 15` (LeidenOrder) |
| General purpose | `-o 7` (HUBCLUSTERDBG) |
| Large graph | `-o 16` (LeidenCSR) |
| Baseline | `-o 0` (no reordering) |

### How do I know which algorithm is best for my graph?

Run multiple algorithms and compare:

```bash
for algo in 0 7 14 15 17; do
    echo "=== Algorithm $algo ==="
    ./bench/bin/pr -f graph.el -s -o $algo -n 3
done
```

Or use AdaptiveOrder (`-o 14`) to auto-select.

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
   ./bench/bin/converter -f graph.el -s -b graph.sg
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

### What are RabbitOrder variants?

RabbitOrder (algorithm 8) has two variants:

| Variant | Command | Description |
|---------|---------|-------------|
| `csr` | `-o 8` or `-o 8:csr` | Native CSR implementation (default) |
| `boost` | `-o 8:boost` | Original Boost-based implementation |

**Recommendations:**
- Use `csr` (default) - faster, no external dependencies
- Use `boost` only if you need the original implementation behavior
- The `boost` variant requires Boost 1.58.0: `--install-boost`

### How does AdaptiveOrder work?

Detects communities via Leiden, computes features (15 linear + 3 quadratic), uses ML perceptron to select best algorithm per community with safety checks. See [[AdaptiveOrder-ML]].

### What is the training pipeline?

4-stage process: multi-restart perceptrons → variant pre-collapse → regret-aware grid search → save. Validate with `python3 scripts/eval_weights.py`. See [[Perceptron-Weights]].

### Is there a single best algorithm?

LeidenCSR was selected for 99.5% of subcommunities in C++ validation. As a single algorithm, it achieves 2.9% median regret. Recommended variant: `graphbrew`.

### What are the quadratic cross-terms?

`w_dv_x_hub` (power-law), `w_mod_x_logn` (large modular graphs), `w_pf_x_wsr` (uniform+cache). See [[AdaptiveOrder-ML#features-used]].

### Why are some perceptron weights 0?

Cache simulation was skipped, features weren't computed, or no benchmark data. Fix: `--train --size small` (full pipeline). See [[Perceptron-Weights#troubleshooting]].

### How do I validate trained weights?

```bash
python3 scripts/eval_weights.py  # Simulates C++ scoring, reports accuracy/regret
```

See [[Python-Scripts#-eval_weightspy---weight-evaluation--c-scoring-simulation]].

### Where are the trained weights saved?

`scripts/weights/active/type_N.json` (per-cluster) + `type_registry.json` (graph→type map). Loading priority: env var → best type match → fallback defaults. See [[Perceptron-Weights#weight-file-location]].

### What's the difference between LeidenOrder and LeidenCSR?

- **LeidenOrder (15)**: Baseline reference using GVE-Leiden external library (requires CSR→DiGraph conversion)
- **LeidenCSR (16)**: Production pure-Leiden implementation, CSR-native (default: `gveopt2`, fastest + best quality)

LeidenCSR reimplements Leiden natively on CSR (zero-copy), achieving equivalent kernel quality but **28–95× faster reorder times**. LeidenOrder is kept as a baseline to measure this improvement.

For per-community reordering (e.g., RabbitOrder within each community), use **GraphBrewOrder (12)**.

### When should I use DBG vs HUBCLUSTER?

| Algorithm | Best For |
|-----------|----------|
| DBG | Power-law graphs with clear hot/cold separation |
| HUBCLUSTER | Graphs where hubs connect to each other |
| HUBCLUSTERDBG | Combines both - good general choice |

---

## Troubleshooting

See [[Troubleshooting]] for detailed solutions. Quick answers:

- **Segfault**: Check file exists, format correct (vertices start at 0), sufficient RAM
- **Results vary**: Normal — use `-n 10`, disable frequency scaling, use `numactl`
- **No speedup**: Not all graphs benefit — try AdaptiveOrder or different algorithm
- **Python issues**: Need Python 3.8+ (no pip required for core scripts)

---

## Development Questions

### How do I add a new reordering algorithm?

See [[Adding-New-Algorithms]] for a complete guide:
1. Add enum value in `reorder_types.h`
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
