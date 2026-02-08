# Frequently Asked Questions (FAQ)

Common questions and answers about GraphBrew.

---

## General Questions

### What is GraphBrew?

GraphBrew is a graph processing benchmark framework that combines:
- **18 vertex reordering algorithms** (IDs 0-17) for cache optimization
- **6 benchmarks** (PageRank, BFS, CC, SSSP, BC, TC) — 5 run by default, TC binary available separately
- **ML-powered algorithm selection** via AdaptiveOrder
- **Leiden community detection** integration

### Who should use GraphBrew?

- **Researchers** studying graph algorithms and cache optimization
- **Engineers** optimizing graph processing pipelines
- **Students** learning about graph algorithms
- **Data scientists** working with network data

### What makes GraphBrew different?

1. **Comprehensive**: 18 reordering algorithms in one framework
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
| Large graph | `-o 17` (LeidenCSR) |
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

1. Detects communities using Leiden
2. Computes features for each community (15 linear + 3 quadratic cross-terms)
3. Uses ML perceptron to select best algorithm per community
4. Applies safety checks (OOD guardrail, ORIGINAL margin fallback)
5. Applies different reorderings to different parts of the graph

See [[AdaptiveOrder-ML]] for details.

### What is the training pipeline?

The `compute_weights_from_results()` function trains weights in 4 stages:
1. **Multi-restart perceptrons** (5 restarts × 800 epochs per benchmark, z-score normalized)
2. **Variant pre-collapse** (keep highest-bias variant per base algorithm)
3. **Regret-aware grid search** for benchmark multipliers (30 iterations × 32 values)
4. **Save to `type_0.json`** with metadata

Validate with `python3 scripts/eval_weights.py` — simulates C++ scoring and reports accuracy/regret.

### What is regret-aware optimization?

When optimizing per-benchmark multipliers, the pipeline jointly maximizes `(accuracy, −mean_regret)`. This prevents degenerate solutions where always predicting the same algorithm gives high accuracy but poor real-world performance. The grid search evaluates 30 random multiplier combinations across 32 log-spaced values.

### Is there a single best algorithm?

C++ validation on 47 graphs showed that **LeidenCSR** was selected for 99.5% of subcommunities (8,631/8,672). As a single algorithm, it achieves 2.9% median regret — very close to optimal per-community selection. LeidenCSR with `gveopt2` variant is recommended for most use cases.

### What is the OOD guardrail?

The **Out-of-Distribution (OOD) guardrail** prevents AdaptiveOrder from making bad predictions on unfamiliar graphs. If a graph's features are too far from any trained type centroid (Euclidean distance > 1.5 in normalized 7D space), the system returns ORIGINAL instead of risking a wrong algorithm choice.

### What is convergence-aware scoring?

For iterative algorithms like **PageRank** and **SSSP**, the perceptron adds a convergence bonus (`w_fef_convergence × forward_edge_fraction`) that rewards orderings where edges point to higher-numbered vertices. This captures how edge direction affects convergence speed, without affecting non-iterative benchmarks.

### What are the quadratic cross-terms?

The perceptron includes 3 non-linear feature interactions:
- `w_dv_x_hub`: degree_variance × hub_concentration (power-law indicator)
- `w_mod_x_logn`: modularity × log₁₀(nodes) (large vs small modular graphs)
- `w_pf_x_wsr`: packing_factor × log₂(working_set_ratio+1) (uniform-degree + cache pressure)

These capture patterns that linear features alone cannot express.

### Why are some perceptron weights 0 or 1.0?

Weight fields remain at 0 or default 1.0 when:
1. **Cache simulation was skipped** (`--skip-cache`) - no cache impact data
2. **Graph features weren't computed** - no topology metrics
3. **No benchmark data** - no per-benchmark weights

**Fix**: Run the comprehensive training mode:
```bash
python3 scripts/graphbrew_experiment.py --train --size small --max-graphs 5 --trials 2
```

This populates all fields including `cache_l1/l2/l3_impact`, `w_clustering_coeff`, `w_diameter`, `benchmark_weights`, and the new features (`w_packing_factor`, `w_forward_edge_fraction`, `w_working_set_ratio`, quadratic cross-terms).

### How do I check if my weights are overfitting?

Use Leave-One-Graph-Out (LOGO) cross-validation:

```python
from scripts.lib.weights import cross_validate_logo
result = cross_validate_logo(benchmark_results, graph_features, type_registry)
print(f"LOGO accuracy: {result['accuracy']:.1%}")
print(f"Overfitting score: {result['overfitting_score']:.2f}")
```

An overfitting score > 0.2 means the model may not generalize well. Try adding more training graphs or increasing L2 regularization.

### How do I validate trained weights without recompiling C++?

Use `eval_weights.py`:
```bash
python3 scripts/eval_weights.py
```

This trains weights from your latest benchmark data, simulates C++ `scoreBase() × benchmarkMultiplier()` scoring, and reports accuracy/regret metrics. Current results: 46.8% accuracy, 2.6% base-aware median regret.

### Why isn't parse_adaptive_output working?

The `parse_adaptive_output()` function in `lib/analysis.py` expects C++ AdaptiveOrder output in the format:
```
Community 0: 12345 nodes, 67890 edges -> LeidenCSR
```

If you see parsing failures, ensure your C++ binary is up to date. The parser also supports the legacy format (`Community N: algo=Algorithm, nodes=X, edges=Y`) as a fallback.

### What is L2 regularization / weight decay?

After each gradient update, all feature weights (`w_*`, `cache_*`) are multiplied by `(1.0 - 1e-4)`. This prevents weights from growing unboundedly during long training runs and improves generalization to unseen graphs.

### Where are the trained weights saved?

Weights are saved using an auto-clustering system:

**Auto-clustered type weights (primary):**
```
scripts/weights/active/
├── type_registry.json    # Maps graph names → type IDs + cluster centroids
├── type_0.json           # Cluster 0 weights
├── type_1.json           # Cluster 1 weights
└── type_N.json           # Additional clusters as needed
```

**Loading priority (C++ runtime):**
1. Environment variable `PERCEPTRON_WEIGHTS_FILE` (if set)
2. Best matching type file via Euclidean distance to centroids (e.g., `type_0.json`)
3. Semantic type fallback (if type files don't exist)
4. Hardcoded defaults

### How does graph type matching work?

The auto-clustering system:
1. **Extracts 7 features** per graph: modularity, log_nodes, log_edges, avg_degree, degree_variance, hub_concentration, clustering_coefficient
2. **Clusters similar graphs** using Euclidean distance to centroids
3. **Stores centroids** in `type_registry.json`
4. **At runtime**, finds best matching cluster based on feature similarity

Properties are computed during `--train` Phase 0 and cached in `results/graph_properties_cache.json`.

### What's the difference between LeidenOrder and LeidenCSR?

| Algorithm | Approach |
|-----------|----------|
| LeidenOrder (15) | Basic Leiden → contiguous community ordering |
| LeidenDendrogram (16) | Leiden + dendrogram-based traversal variants |
| LeidenCSR (17) | Optimized Leiden with multiple variants (see `LEIDEN_CSR_VARIANTS` in `scripts/lib/utils.py`) |

LeidenCSR with `gveopt2` or `gvefast` variant is recommended for large graphs.

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
# Core scripts need only Python 3.8+ standard library (no pip install needed)
# Optional: Install for extended visualization
pip install numpy matplotlib pandas

# Check Python version
python3 --version  # Need 3.8+

# Check path and test script
python3 scripts/graphbrew_experiment.py --help
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
