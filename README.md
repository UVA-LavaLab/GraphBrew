[![Build Status](https://app.travis-ci.com/UVA-LavaLab/GraphBrew.svg?branch=main)](https://app.travis-ci.com/UVA-LavaLab/GraphBrew)
[![Wiki](https://img.shields.io/badge/📚_Wiki-Documentation-blue?style=flat)](https://github.com/UVA-LavaLab/GraphBrew/wiki)

[<p align="center"><img src="./docs/figures/logo.svg" width="200" ></p>](#graphbrew)

# GraphBrew <img src="./docs/figures/logo_left.png" width="50" align="center">

A graph reordering and benchmarking framework built on the [GAP Benchmark Suite (GAPBS)](https://github.com/sbeamer/gapbs). GraphBrew reorders graph vertices to improve cache locality and speed up graph algorithms — with **17 reordering algorithms**, an **ML-based adaptive selector**, and a **one-click experiment pipeline**.

> **📖 Full documentation:** [Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki) · [Quick Start](https://github.com/UVA-LavaLab/GraphBrew/wiki/Quick-Start) · [Command-Line Reference](https://github.com/UVA-LavaLab/GraphBrew/wiki/Command-Line-Reference)

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/UVA-LavaLab/GraphBrew.git
cd GraphBrew
make all

# Run PageRank with AdaptiveOrder (ML-selected best algorithm)
./bench/bin/pr -g 20 -o 14

# Run BFS with LeidenCSR reordering on a graph file
./bench/bin/bfs -f graph.mtx -o 16
```

### Build with RabbitOrder (optional)

Requires Boost, libnuma, and google-perftools. See [Installation Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Installation) for details.

```bash
make RABBIT_ENABLE=1 all
```

---

## Reordering Algorithms

GraphBrew provides 17 reordering strategies. Use `-o <id>` to select one (or chain multiple with `-o <id1> -o <id2>`):

| ID | Algorithm | Description |
|----|-----------|-------------|
| 0 | ORIGINAL | No reordering |
| 1 | RANDOM | Random permutation |
| 2 | SORT | Sort by degree |
| 3 | HUBSORT | Hub-based sorting |
| 4 | HUBCLUSTER | Hub-score clustering |
| 5 | DBG | Degree-based grouping |
| 6 | HUBSORTDBG | HubSort + DBG |
| 7 | HUBCLUSTERDBG | HubCluster + DBG |
| 8 | RABBITORDER | Community clustering (variants: `csr` / `boost`) |
| 9 | GORDER | Window-based BFS ordering |
| 10 | CORDER | Workload-balanced reordering |
| 11 | RCM | Reverse Cuthill-McKee |
| 12 | GRAPHBREWORDER | Leiden clustering + configurable per-community order |
| 13 | MAP | Load ordering from file (`-o 13:mapping.lo`) |
| 14 | **ADAPTIVEORDER** | ML perceptron — automatically picks the best algorithm ⭐ |
| 15 | LEIDENORDER | Leiden via GVE-Leiden library (`15:resolution`) — baseline reference |
| 16 | LEIDENCSR | Fast CSR-native Leiden (`16:variant:resolution:passes`) — default: `gveopt2` ⭐ |

**LeidenCSR (16) variants:** `gveopt2` (default — fastest + best quality), `gve`, `gveopt`, `fast`, `modularity`, `gverabbit`, `dfs`, `bfs`, `hubsort`, `faithful`. See [Reordering Algorithms Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Reordering-Algorithms) for the full list.

### Which Algorithm Should I Use?

| Graph Type | Recommended | Why |
|------------|-------------|-----|
| Social networks | `12` or `14` | Best community detection + cache locality |
| Web graphs | `12:hrab` or `12` | Hybrid Leiden+Rabbit for best locality |
| Road networks | `11` or `9` | BFS-based approaches for sparse graphs |
| Unknown / mixed | `14` | Let the ML perceptron decide |

---

## One-Click Experiment Pipeline

Run the complete benchmark + training workflow with one command:

```bash
# Install Python dependencies
pip install -r scripts/requirements.txt

# Full pipeline: download graphs → build → benchmark → train ML weights
python3 scripts/graphbrew_experiment.py --train --all-variants --size medium --auto --trials 5
```

| Parameter | Description |
|-----------|-------------|
| `--train` | Train perceptron weights (runs the complete pipeline) |
| `--full` | Run full evaluation pipeline (no training) |
| `--all-variants` | Test all algorithm variants |
| `--size SIZE` | Graph category: `small` (62 MB), `medium` (1.1 GB), `large` (25 GB), `xlarge` (63 GB), `all` (89 GB) |
| `--auto` | Auto-detect RAM/disk limits |
| `--trials N` | Benchmark trials (default: 2) |
| `--quick` | Test only key algorithms (faster) |
| `--brute-force` | Compare adaptive selection vs all 17 algorithms |
| `--download-only` | Download graphs without running benchmarks |

Results are saved to `./results/`. Trained weights go to `./scripts/weights/active/`.

```bash
# See all options
python3 scripts/graphbrew_experiment.py --help
```

> 📖 See [Running Benchmarks Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Running-Benchmarks) for advanced workflows.

---

## Graph Benchmarks

Built on [GAPBS](https://github.com/sbeamer/gapbs), GraphBrew includes these benchmarks:

| Benchmark | Algorithm |
|-----------|-----------|
| `pr` | PageRank |
| `bfs` | Breadth-First Search (direction optimized) |
| `cc` | Connected Components (Afforest) |
| `cc_sv` | Connected Components (Shiloach-Vishkin) |
| `sssp` | Single-Source Shortest Paths |
| `bc` | Betweenness Centrality |
| `tc` | Triangle Counting |
| `pr_spmv` | PageRank (SpMV variant) |

```bash
# Run a single benchmark
make run-bfs

# Run with parameters
./bench/bin/pr -f graph.mtx -n 16 -o 12

# Generate a reordered graph
./bench/bin/converter -f graph.mtx -p reordered.mtx -o 16
```

> 📖 See [Graph Benchmarks Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Graph-Benchmarks) for details.

---

## Supported Graph Formats

GraphBrew can load graphs in these formats:

| Extension | Format |
|-----------|--------|
| `.el` | Edge list (node1 node2) |
| `.wel` | Weighted edge list |
| `.mtx` | Matrix Market |
| `.gr` | DIMACS |
| `.graph` | Metis |
| `.sg` / `.wsg` | Serialized (pre-built via `converter`) |

Graphs can also be generated synthetically:
- `-g 20` — Kronecker graph with 2²⁰ vertices (Graph500)
- `-u 20` — Uniform random graph with 2²⁰ vertices

> 📖 See [Supported Graph Formats Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Supported-Graph-Formats) for details.

---

## Prerequisites

- **OS:** Ubuntu 22.04+ (or any Linux with GCC ≥ 9)
- **Compiler:** `g++` with C++17 and OpenMP support
- **Build:** `make`
- **Python:** 3.8+ (for experiment scripts)

Optional (for RabbitOrder): Boost 1.58+, libnuma, google-perftools.

> 📖 See [Installation Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Installation) for full setup instructions including Boost.

---

## Project Layout

```
bench/
├── src/          # Benchmark source files (bc.cc, bfs.cc, pr.cc, ...)
├── src_sim/      # Cache simulation variants
├── bin/          # Compiled binaries
└── include/
    ├── graphbrew/    # Reordering algorithms & partitioning
    ├── external/     # Bundled libraries (GAPBS, RabbitOrder, GOrder, COrder, Leiden)
    └── cache_sim/    # Cache simulation headers

scripts/
├── graphbrew_experiment.py   # Main experiment orchestration
├── lib/                      # Core Python modules
├── weights/                  # Trained ML weights
└── test/                     # Test suite

results/                      # Benchmark outputs, graph features, mappings
```

> 📖 See [Code Architecture Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki/Code-Architecture) for the full layout.

---

## Testing

```bash
pip install -r scripts/requirements.txt
pytest scripts/test -q

# Topology verification (ensures reordering preserves graph structure)
make test-topology
```

---

## Developer Tooling

```bash
make lint-includes              # Check for legacy include paths
make help                       # Show all Make targets
make help-pr                    # Show parameters for a specific benchmark
```

---

## How to Cite

If you use GraphBrew in your research, please cite:

- S. Beamer, K. Asanović, D. Patterson, "The GAP Benchmark Suite," arXiv:1508.03619, 2017.
- J. Arai, H. Shiokawa, T. Yamamuro, M. Onizuka, S. Iwamura, "Rabbit Order: Just-in-time Parallel Reordering for Fast Graph Analysis."
- P. Faldu, J. Diamond, B. Grot, "A Closer Look at Lightweight Graph Reordering," arXiv:2001.08448, 2020.
- S. Sahu, "GVE-Leiden: Fast Leiden Algorithm for Community Detection in Shared Memory Setting," arXiv:2312.13936, 2024.
- V. A. Traag, L. Waltman, N. J. van Eck, "From Louvain to Leiden: guaranteeing well-connected communities," Sci Rep 9, 5233, 2019.
- H. Wei, J. X. Yu, C. Lu, X. Lin, "Speedup Graph Processing by Graph Ordering," SIGMOD 2016.
- Y. Chen, Y.-C. Chung, "Workload Balancing via Graph Reordering on Multicore Systems," IEEE TPDS, 2021.

---

## License

See [LICENSE](LICENSE) for details.
