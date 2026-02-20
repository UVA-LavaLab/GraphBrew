# Code Architecture

Understanding the GraphBrew codebase structure for developers.

## Directory Structure

```
GraphBrew/
├── bench/                    # Core C++ benchmark code
│   ├── bin/                  # Compiled binaries
│   ├── bin_sim/              # Cache simulation binaries
│   ├── include/              # Header libraries
│   │   ├── graphbrew/        # 📦 GraphBrew extensions
│   │   │   ├── graphbrew.h   # Umbrella header
│   │   │   ├── reorder/      # Reordering algorithms (~18,100 lines)
│   │   │   └── partition/    # Partitioning (trust.h, cagra/popt.h)
│   │   ├── external/         # External libraries (bundled)
│   │   │   ├── gapbs/        # Core GAPBS runtime (builder.h ~3,751 lines)
│   │   │   ├── rabbit/       # RabbitOrder
│   │   │   ├── gorder/       # GOrder
│   │   │   ├── corder/       # COrder
│   │   │   └── leiden/       # GVE-Leiden
│   │   └── cache_sim/        # Cache simulation headers
│   ├── src/                  # Benchmark source files
│   ├── src_sim/              # Cache simulation sources
│   └── backups/              # Backup files
│
├── scripts/                  # Python tools (~28,600 lines)
│   ├── graphbrew_experiment.py  # Main orchestration
│   ├── lib/                     # Shared modules
│   ├── benchmark/               # Benchmark shell scripts
│   └── examples/                # Usage examples
│
├── scripts/test/             # Pytest suite
│   ├── graphs/               # Sample graphs
│   └── data/                 # Test data
│
├── docs/                     # Documentation
│   └── figures/              # Images
│
└── wiki/                     # This wiki
```

---

## Core Components

### GraphBrew Extensions (bench/include/graphbrew/)

| Module | Purpose |
|--------|---------|
| graphbrew.h | Umbrella header (includes everything) |
| reorder/ | Reordering algorithm implementations |
| partition/ | Partitioning (trust.h, cagra/popt.h) |

### External Libraries (bench/include/external/)

| Module | Notes |
|--------|-------|
| gapbs/ | Core GAPBS runtime (builder.h, graph.h, etc.) |
| rabbit/ | RabbitOrder community clustering |
| gorder/ | GOrder implementation |
| corder/ | COrder (cache-aware ordering) |
| leiden/ | Leiden community detection |

The foundation is built on the GAP Benchmark Suite with extensions.

#### Key Files

| File | Lines | Purpose |
|------|-------|---------|
| [graph.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/graph.h) | ~729 | CSRGraph class, core data structure |
| [builder.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/builder.h) | ~3,751 | Graph loading and reordering dispatcher |
| [benchmark.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/benchmark.h) | ~256 | Benchmark harness |
| [command_line.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/command_line.h) | ~533 | CLI parsing |
| [pvector.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/pvector.h) | ~204 | Parallel-friendly vector |
| [timer.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/timer.h) | ~50 | High-resolution timing |

#### Partitioning Modules

| File | Lines | Purpose |
|------|-------|---------|
| `partition/cagra/popt.h` | ~892 | `graphSlicer`, `MakeCagraPartitionedGraph`, cache optimization (P-OPT) |
| `partition/trust.h` | ~751 | `TrustPartitioner` class for triangle-count partitioning |

> **Cache vs Cagra:** Cache **simulation** lives in `bench/include/cache_sim/` (`cache_sim.h`, `graph_sim.h`). Cagra **partitioning** helpers live in `bench/include/graphbrew/partition/cagra/` (`popt.h`). See `docs/INDEX.md` and folder READMEs for a quick map.

#### Reorder Module (bench/include/graphbrew/reorder/)

The reorder module is a modular header library with standalone template functions. It follows an include hierarchy where `reorder_types.h` is the base, and specialized headers extend it.

```
reorder/
├── reorder_types.h      # Base: types, perceptron, feature computation (~5,052 lines)
├── reorder_basic.h      # Original, Random, Sort (algo 0-2) (~324 lines)
├── reorder_hub.h        # HubSort, HubCluster, DBG variants (algo 3-7) (~641 lines)
├── reorder_rabbit.h     # RabbitOrder native CSR (algo 8) (~1,141 lines)
├── reorder_classic.h    # GOrder, COrder, RCMOrder dispatch (algo 9-11) (~521 lines)
├── reorder_gorder.h     # GOrder CSR variants: serial (-o 9:csr) + parallel (-o 9:fast) (~932 lines)
├── reorder_rcm.h        # RCM BNF variant (-o 11:bnf) (~645 lines)
├── reorder_adaptive.h   # ML-based selection (algo 14) (~802 lines)
├── reorder_graphbrew.h  # GraphBrew + Leiden unified reordering (algo 12, 15) (~7,490 lines)
└── reorder.h            # Main dispatcher (~570 lines)
```

**Total: ~18,118 lines**

| File | Lines | Purpose |
|------|-------|---------|
| `reorder_graphbrew.h` | ~7,490 | GraphBrew + Leiden unified reordering framework (algo 12, 15) |
| `reorder_types.h` | ~5,052 | Common types, perceptron model, `EdgeList`, threshold functions, `GetLLCSizeBytes()`, `getAlgorithmNameMap()` (~16 base names), `lookupAlgorithm()`, `ResolveVariantSelection()` |
| `reorder_rabbit.h` | ~1,141 | RabbitOrder CSR native implementation (auto-adaptive resolution) |
| `reorder_gorder.h` | ~932 | GOrder CSR variants: serial greedy (-o 9:csr) + parallel batch (-o 9:fast) |
| `reorder_adaptive.h` | ~802 | `AdaptiveConfig`, ML-based per-community algorithm selection |
| `reorder_rcm.h` | ~645 | RCM BNF variant: CSR-native BNF start + deterministic parallel CM BFS |
| `reorder_hub.h` | ~641 | Hub-based algorithms (DBG, HubSort, HubCluster) |
| `reorder.h` | ~570 | Main dispatcher, `ApplyBasicReorderingStandalone` |
| `reorder_classic.h` | ~521 | Classic algorithms (GOrder, COrder, RCM dispatch) |
| `reorder_basic.h` | ~324 | Basic algorithms (Original, Random, Sort) |

**Key Utilities in reorder_types.h:**

- `PerceptronWeights` / `TypeRegistry` — ML scoring & graph clustering (see [[AdaptiveOrder-ML]])
- `SampledDegreeFeatures` — 8-feature topology vector: degree_variance, hub_concentration, avg_degree, clustering_coeff, estimated_modularity, packing_factor, forward_edge_fraction, working_set_ratio
- `ComputeSampledDegreeFeatures()` — Auto-scaled sampling (max(5000, min(√N, 50000))) for fast feature extraction
- `GetLLCSizeBytes()` — LLC detection (sysconf on Linux, 30MB fallback) for working_set_ratio
- `getAlgorithmNameMap()` — ~16-entry UPPERCASE base-name→enum mapping; variant names resolved dynamically by `ResolveVariantSelection()` via prefix matching (see [[Command-Line-Reference]])

**Key Configs:**

| Struct | Header | Key Fields |
|--------|--------|------------|
| `AdaptiveConfig` | `reorder_adaptive.h` | max_depth, resolution, min_recurse_size, mode (0=per-community, 1=full-graph) |
| `GraphBrewConfig` | `reorder_graphbrew.h` | algorithm, ordering, aggregation, resolution, finalAlgoId, recursiveDepth, subAlgoId |
| `ReorderConfig` | `reorder_types.h` | Unified config: resolutionMode(AUTO), tolerance(1e-2), maxIterations(10), maxPasses(10), ordering(HIERARCHICAL) |

All configs parse from CLI options via `FromOptions()`. Defaults are centralized constants in `reorder_types.h` (see [[AdaptiveOrder-ML#command-line-format]]).

> ⚠️ Use `graphbrew::leiden::DEFAULT_RESOLUTION` or `adaptive::DEFAULT_RESOLUTION` explicitly — they are separate namespaces.

#### graph.h - CSRGraph Class

```cpp
template <class NodeID_ = int32_t, class DestID_ = NodeID_,
          class WeightT_ = NodeID_>
class CSRGraph {
 public:
  // Accessors
  int64_t num_nodes() const;
  int64_t num_edges() const;
  int64_t num_edges_directed() const;
  
  // Degree queries
  int64_t out_degree(NodeID_ n) const;
  int64_t in_degree(NodeID_ n) const;
  
  // Neighborhood iteration
  Neighborhood out_neigh(NodeID_ n) const;
  Neighborhood in_neigh(NodeID_ n) const;
  
 private:
  int64_t num_nodes_;
  int64_t num_edges_;
  DestID_** out_index_;  // CSR row pointers
  DestID_* out_neighbors_; // CSR column indices
};
```

#### builder.h - Graph Construction

`BuilderBase` is the template class; benchmarks use the typedef `Builder` from `benchmark.h`:

```cpp
// In benchmark.h
typedef BuilderBase<NodeID, NodeID, WeightT> Builder;

// BuilderBase class definition
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
class BuilderBase {
 public:
  BuilderBase(const CLBase &cli);
  
  // Main entry point
  CSRGraph<NodeID_, DestID_, invert> MakeGraph();
  
 private:
  // Graph loading
  EdgeList ReadEdgeList(string filename);
  CSRGraph MakeGraphFromEL(EdgeList& el);
  
  // Reordering implementation
  void GenerateMapping(CSRGraph& g, pvector<NodeID_>& new_ids, 
                       ReorderingAlgo algo, ...);
};
```

---

### Reordering Algorithms

#### Hub-Based (bench/include/graphbrew/reorder/reorder_hub.h)

All hub functions share the same template signature:
```cpp
template<typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void Generate{Algorithm}MappingStandalone(const CSRGraph<...>& g,
    pvector<NodeID_>& new_ids, bool useOutdeg);
```

| Function | Strategy |
|----------|----------|
| `GenerateDBGMapping` | Groups vertices by log₂(degree) |
| `GenerateHubSortMapping` | Sorts by degree, high-degree first |
| `GenerateHubClusterMapping` | Clusters hot vertices together |
| `GenerateHubClusterDBGMapping` | Combines hub clustering with DBG |

**Edge Case Guard:** All reordering functions check `num_nodes == 0` to prevent FPE on empty subgraphs (important for GraphBrewOrder on Kronecker graphs).

#### Community-Based

| Algorithm | File | Method |
|-----------|------|--------|
| RabbitOrder | `graphbrew/reorder/reorder_rabbit.h` | Native CSR community detection + recursion |
| Gorder | `external/gorder/GoGraph.h` | Window optimization (default); CSR serial + parallel batch: `reorder_gorder.h` |
| Corder | `external/corder/global.h` | Cache-aware ordering |
| RCM | `graphbrew/reorder/reorder_classic.h` | Cuthill-McKee dispatch (default: GoGraph; BNF: `reorder_rcm.h`) |

#### Leiden-Based (bench/include/graphbrew/reorder/reorder_graphbrew.h)

```cpp
// LeidenOrder - Leiden community detection via GVE-Leiden (baseline)
template<typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateLeidenMapping(
    const CSRGraph<NodeID_, DestID_, invert>& g, pvector<NodeID_>& new_ids,
    const ReorderingOptions& opts);
```

The Leiden community detection algorithm itself is in `bench/include/external/leiden/leiden.hxx`:

```cpp
// Core Leiden algorithm (GVE-Leiden)
inline auto leidenStatic(RND& rnd, const G& x, const LeidenOptions& o={});
inline auto leidenStaticOmp(RND& rnd, G& x, const LeidenOptions& o={});
```

---

### Benchmarks (bench/src/)

#### Standard Pattern

All benchmarks follow this pattern:

```cpp
int main(int argc, char* argv[]) {
  // 1. Parse command line
  CLApp cli(argc, argv, "benchmark_name");
  if (!cli.ParseArgs()) return -1;
  
  // 2. Build graph (with optional reordering)
  Builder b(cli);
  Graph g = b.MakeGraph();
  
  // 3. Run benchmark with timing
  ResultType* result;
  BenchmarkKernel(cli, g, [&](const Graph& g) {
    result = Algorithm(g);
    return result;
  });
  
  // 4. Verify and output
  if (cli.verify()) VerifyResult(g, result);
  PrintResult(g, result);
  
  return 0;
}
```

#### Benchmark Files

| File | Algorithm | Key Function |
|------|-----------|--------------|
| `pr.cc` | PageRank | `PageRankPullGS()` |
| `pr_spmv.cc` | PageRank (SpMV) | `PageRankPull()` |
| `bfs.cc` | BFS | `DOBFS()` |
| `cc.cc` | Connected Components | `Afforest()` |
| `cc_sv.cc` | Connected Components (SV) | `ShiloachVishkin()` |
| `sssp.cc` | Shortest Paths | `DeltaStep()` |
| `bc.cc` | Betweenness | `Brandes()` |
| `tc.cc` | Triangles | `OrderedCount()` |

---

### Python Scripts (scripts/)

See [[Python-Scripts]] for full documentation of the Python tooling.

Key entry points:
- `graphbrew_experiment.py` — Main orchestration (~2,680 lines)
- `lib/perceptron.py` — ML weight experimentation
- `lib/adaptive_emulator.py` — C++ logic emulation
- `lib/eval_weights.py` — Weight evaluation & accuracy reporting
- `lib/` — 30 reusable modules (~22,400 lines total)

**Unified Naming Convention (SSOT):** All Python modules use five SSOT functions from `lib/utils.py`:

| Function | Purpose | Example |
|----------|---------|---------|
| `canonical_algo_key(algo_id, variant)` | Canonical name for weights/filenames/JSON | `canonical_algo_key(12, "leiden")` → `"GraphBrewOrder_leiden"` |
| `algo_converter_opt(algo_id, variant)` | C++ `-o` argument | `algo_converter_opt(8, "boost")` → `"8:boost"` |
| `canonical_name_from_converter_opt(opt)` | Reverse: `-o` string → canonical name | `canonical_name_from_converter_opt("12:leiden")` → `"GraphBrewOrder_leiden"` |
| `chain_canonical_name(converter_opts)` | Multi-step chain name | `chain_canonical_name("-o 2 -o 8:csr")` → `"SORT+RABBITORDER_csr"` |
| `get_algo_variants(algo_id)` | Variant tuple (or `None`) | `get_algo_variants(12)` → `("leiden", "rabbit", "hubcluster")` |

**Chained Orderings:** `CHAINED_ORDERINGS` is auto-populated at module load from `_CHAINED_ORDERING_OPTS` via `chain_canonical_name()`. These are pregeneration-only (not used in perceptron training). Each entry is a `(canonical_name, converter_opts)` tuple. Current chains: `SORT+RABBITORDER_csr`, `SORT+RABBITORDER_boost`, `HUBCLUSTERDBG+RABBITORDER_csr`, `SORT+GraphBrewOrder_leiden`, `DBG+GraphBrewOrder_leiden`.

**Variant Registry:** `_VARIANT_ALGO_REGISTRY` maps algo IDs 8, 11, 12 to `(prefix, variants, default)` tuples. GOrder variants (9: default/csr/fast) are tracked separately in `GORDER_VARIANTS` but share a single perceptron weight (they produce equivalent orderings).

See [[Configuration-Files#unified-algorithm-naming-scriptslibutilspy]].

---

## Data Flow

### Graph Loading Pipeline

```
Input File → Reader → EdgeList → CSRGraph
     ↓           ↓         ↓          ↓
   .el/.mtx   ParseLine  Edges    Compressed
```

### Reordering Pipeline

```
CSRGraph → Analyzer → Algorithm → Mapping → RelabeledGraph
    ↓          ↓          ↓          ↓            ↓
  Input    Features   Compute    NodeID[]     Output
```

### Benchmark Pipeline

```
CSRGraph → Warmup → Trials → Timer → Results → Output
    ↓         ↓        ↓        ↓        ↓         ↓
  Input    Cache    N runs  Measure  Verify    Print
```

---

## Key Data Structures

| Type | Header | Purpose |
|------|--------|---------|
| `CSRGraph<NodeID_, DestID_, WeightT_>` | `graph.h` | Core graph: CSR row pointers + column indices |
| `pvector<T>` | `pvector.h` | Parallel-friendly aligned vector |
| `SlidingQueue<T>` | `sliding_queue.h` | Lock-free queue for BFS frontier |
| `Bitmap` | `bitmap.h` | Efficient bit vector (set/get/reset, atomic ops) |

**Parallelization:** OpenMP `parallel for` with `reduction` for sums, `atomic` for counters, and thread-local buffers merged via `critical` sections.

**CSR layout:** Nodes index into a flat neighbor array — sequential iteration is cache-friendly, and reordering places community members in adjacent positions.

---

## Configuration & Data Locations

JSON config: specify `graphs`, `benchmarks`, `algorithms`, `trials`, and `options` (symmetrize, verify). See [[Configuration-Files]] for format.

Weight files: `results/weights/type_*/weights.json` (see [[Perceptron-Weights]]). Results: `results/graphs/`, `results/logs/`, `results/mappings/` (see [[Python-Scripts#output-structure]]).

---

## Error Handling

### Common Patterns

```cpp
// File not found
ifstream ifs(filename);
if (!ifs.is_open()) {
  cerr << "Error: Cannot open " << filename << endl;
  exit(1);
}

// Invalid algorithm
if (algo < 0 || algo > MAX_ALGO) {
  cerr << "Error: Unknown algorithm " << algo << endl;
  exit(1);
}

// Graceful fallback
try {
  result = ExpensiveOperation();
} catch (const exception& e) {
  cerr << "Warning: " << e.what() << ", using fallback" << endl;
  result = FallbackOperation();
}
```

---

## Extending the Codebase

### Adding a Feature

1. **Header-only**: Add to appropriate `include/` directory
2. **Source file**: Add to `bench/src/`, update Makefile
3. **Python**: Add to `scripts/`, update imports

### Testing Changes

```bash
# Build
make clean && make all

# Quick test
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -n 1

# Full test
make test

# Memory check
valgrind ./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -n 1
```

### Code Style

- **C++**: Follow existing style (2-space indent, K&R braces)
- **Python**: PEP 8, type hints where helpful
- **Comments**: Explain why, not what

---

## Next Steps

- [[Adding-New-Algorithms]] - Add reordering algorithms
- [[Adding-New-Benchmarks]] - Add graph algorithms
- [[Python-Scripts]] - Python tools documentation

---

[← Back to Home](Home) | [Python Scripts →](Python-Scripts)
