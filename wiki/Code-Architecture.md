# Code Architecture

Understanding the GraphBrew codebase structure for developers.

[![GraphBrew pipeline architecture](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v4)

The six numbered cards are the public stage map. Use
[GraphBrew Running Example](GraphBrew-Running-Example) for the exact graph,
mapping, CSR row, and locality values before following the implementation
paths below.

## Directory Structure

```
GraphBrew/
├── bench/                    # Core C++ benchmark code
│   ├── bin/                  # Compiled binaries
│   ├── bin_sim/              # Cache simulation binaries
│   ├── include/              # Header libraries
│   │   ├── graphbrew/        # GraphBrew extensions
│   │   │   ├── graphbrew.h   # Umbrella header
│   │   │   ├── reorder/      # Reordering algorithms
│   │   │   └── partition/    # Partitioning (trust.h, cagra/popt.h)
│   │   ├── external/         # External libraries (bundled)
│   │   │   ├── gapbs/        # Core GAPBS runtime
│   │   │   ├── rabbit/       # RabbitOrder
│   │   │   ├── gorder/       # GOrder
│   │   │   ├── corder/       # COrder
│   │   │   └── leiden/       # GVE-Leiden
│   │   └── cache_sim/        # Cache simulation headers
│   ├── src/                  # Canonical benchmark baselines
│   ├── src_edge/             # Edge-centric benchmark drivers
│   ├── src_gas/              # Natural GAS benchmark drivers
│   └── src_sim/              # Cache simulation sources
│
├── scripts/                  # Python tools
│   ├── graphbrew_experiment.py  # Main orchestration
│   ├── lib/                     # Shared modules (5 sub-packages: core, pipeline, ml, analysis, tools)
│   └── test/                    # Pytest suite
│       ├── graphs/              # Sample graphs
│       └── data/                # Test data
│
├── results/                  # Experiment outputs
│   ├── data/                 # Structured data store
│   │   ├── adaptive_models.json     # Historical offline-model store
│   │   ├── benchmarks.json          # Benchmark database
│   │   └── graph_properties.json    # Graph feature cache
│   ├── graphs/               # Downloaded graphs
│   ├── logs/                 # Run logs
│   └── mappings/             # Node mappings (.lo files)
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

| File | Purpose |
|------|---------|
| [graph.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/graph.h) | CSRGraph class and core graph representation |
| [builder.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/builder.h) | Graph loading and reordering dispatch |
| [benchmark.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/benchmark.h) | Benchmark lifecycle and timing |
| [command_line.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/command_line.h) | CLI parsing |
| [pvector.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/pvector.h) | Parallel-friendly vector |
| [timer.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/timer.h) | Timing utility |

#### Partitioning Modules

| File | Purpose |
|------|---------|
| `partition/cagra/popt.h` | Cagra/GraphIT slicing and P-OPT helpers |
| `partition/trust.h` | TRUST triangle-count partitioning |

> **Cache vs Cagra:** Cache **simulation** lives in `bench/include/cache_sim/` (`cache_sim.h`, `graph_sim.h`, `graph_cache_context.h`). Cagra **partitioning** helpers live in `bench/include/graphbrew/partition/cagra/` (`popt.h`). See `docs/INDEX.md` and folder READMEs for a quick map.
>
> **Graph Cache Context:** `graph_cache_context.h` owns the shared graph and
> property metadata consumed by GRASP, P-OPT, and ECG simulation.

#### Reorder Module (bench/include/graphbrew/reorder/)

The reorder module is a modular header library with standalone template functions. It follows an include hierarchy where `reorder_types.h` is the base, and specialized headers extend it.

```
reorder/
├── reorder_types.h      # Base types, feature computation, legacy model types
├── reorder_basic.h      # Original, Random, Sort (algo 0-2)
├── reorder_hub.h        # HubSort, HubCluster, DBG variants (algo 3-7)
├── reorder_rabbit.h     # RabbitOrder native CSR (algo 8)
├── reorder_classic.h    # GOrder, COrder, RCMOrder dispatch (algo 9-11)
├── reorder_gorder.h     # GOrder CSR variants
├── reorder_rcm.h        # RCM BNF variant
├── reorder_adaptive.h   # deterministic rules + legacy model modes (algo 14)
├── reorder_database.h   # Self-recording + retained offline diagnostics
├── reorder_graphbrew.h  # GraphBrew + Leiden core and COMPOSE path
├── reorder_graphbrew_diagnostics.h # Callable diagnostic orderings
├── reorder_graphbrew_parser.h # GraphBrew token parser
└── reorder.h            # Main dispatcher
```

**Key Utilities in reorder_types.h:**

- `PerceptronWeights` / `GraphType` — retained offline-model research types
- sampled feature structures and deterministic Tier-0 extraction
- `GetLLCSizeBytes()` — LLC detection (sysconf on Linux, 30MB fallback) for working_set_ratio
- `getAlgorithmNameMap()` — base-name lookup; variants are resolved by `ResolveVariantSelection()`

**Key Configs:**

| Struct | Header | Key Fields |
|--------|--------|------------|
| `AdaptiveConfig` | `reorder_adaptive.h` | runtime selection policy, criterion, and explicit reuse |
| `GraphBrewConfig` | `reorder_graphbrew.h` | algorithm, ordering, aggregation, resolution, finalAlgoId, recursiveDepth, subAlgoId |
| `ReorderConfig` | `reorder_types.h` | Unified config: resolutionMode(AUTO), tolerance(1e-2), maxIterations(10), maxPasses(10), ordering(HIERARCHICAL) |

All configs parse from CLI options via `FromOptions()`. Runtime-policy
semantics are documented in [AdaptiveOrder](AdaptiveOrder).

> Note: use `graphbrew::leiden::DEFAULT_RESOLUTION` or
> `adaptive::DEFAULT_RESOLUTION` explicitly — they are separate namespaces.

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
- `graphbrew_experiment.py` — Public experiment orchestrator
- `lib/core/datastore.py` — Versioned raw observations and graph properties
- `lib/pipeline/benchmark.py` — Benchmark execution and timing parsing
- `lib/pipeline/reorder_config.py` — Effective/realized reorder config validation
- `lib/pipeline/suitesparse_catalog.py` — SuiteSparse auto-discovery (ssgetpy)
- `lib/pipeline/cache.py` — Cache simulation

Retained offline-model and parity tools live under `lib/ml/` and
`lib/tools/evaluate_all_modes.py`; they are optional compatibility tools.

#### Adaptive policy compatibility

Algorithm 14 dispatches to an existing reorderer through a registered policy.
It has no intrinsic permutation, so result records must preserve both the
requested policy and the resolved mapping.

Perceptron, decision-tree, hybrid, and kNN code remains under
`scripts/lib/ml/` and the C++ compatibility headers for offline analysis.

**Unified Naming Convention (SSOT):** All Python modules use five SSOT functions from `lib/core/utils.py`:

| Function | Purpose | Example |
|----------|---------|---------|
| `canonical_algo_key(algo_id, variant)` | Canonical name for weights/filenames/JSON | `canonical_algo_key(12, "leiden")` → `"GraphBrewOrder_leiden"` |
| `algo_converter_opt(algo_id, variant)` | C++ `-o` argument | `algo_converter_opt(8, "boost")` → `"8:boost"` |
| `canonical_name_from_converter_opt(opt)` | Reverse: `-o` string → canonical name | `canonical_name_from_converter_opt("12:leiden")` → `"GraphBrewOrder_leiden"` |
| `chain_canonical_name(converter_opts)` | Multi-step chain name | `chain_canonical_name("-o 12:leiden -o 5")` → `"GraphBrewOrder_leiden+DBG"` |
| `get_algo_variants(algo_id)` | Variant tuple (or `None`) | `get_algo_variants(12)` → `("leiden", "rabbit", "hubcluster")` |

**Chained Orderings:** `CHAINED_ORDERINGS` is auto-populated at module load from `_CHAINED_ORDERING_OPTS` via `chain_canonical_name()`. These are explicit pregeneration treatments. Each entry is a `(canonical_name, converter_opts)` tuple.

**Variant Registry:** `_VARIANT_ALGO_REGISTRY` maps algorithm IDs to
canonical names and converter options. Faithful and relaxed variants remain
separate experimental configurations even when historical model tooling
shares metadata.

See [[Python-Scripts]].

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

### Self-Recording Database (v2.1)

C++ benchmark binaries now write directly to `benchmarks.json` and
`graph_properties.json`, eliminating Python as the data-persistence middleman.

```
┌─────────────────────────────────────────────────────────────────┐
│  C++ Self-Recording Pipeline                                     │
│                                                                   │
│  main() → InitSelfRecording(cli.db_dir())                        │
│    ↓  resolves: --db-dir > $GRAPHBREW_DB_DIR > default           │
│    ↓  enables SelfRecordingEnabled() if explicit source found     │
│                                                                   │
│  Builder::MakeGraph()                                             │
│    → ComputeAndPrintGlobalTopologyFeatures()                     │
│      → update_graph_props(props)  [writes graph_properties.json] │
│    → GenerateMapping()                                            │
│      → AppendReorderMetaHint(meta)  [stored for BenchmarkKernel] │
│                                                                   │
│  BenchmarkKernel(cli, g, kernel, stats, verify, name, extractor) │
│    → per-trial TrialResult(time, answer_json)                    │
│    → RunReport(graph, algorithm, benchmark, trials, reorder_meta)│
│    → append_run(report)  [file-locked write to benchmarks.json]  │
│                                                                   │
│  Python sets GRAPHBREW_DB_DIR=results/data/ via os.environ        │
│  (utils.py at module load time)                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key files:**
- `reorder_database.h` — `InitSelfRecording()`, `append_run()`, `update_graph_props()`, `FileLockGuard`
- `benchmark.h` — 7-arg `BenchmarkKernel` overload with `benchmark_name` + `result_extractor`
- `builder.h` — auto-records graph properties and reorder metadata
- `command_line.h` — `--db-dir` / `-D` flag on `CLBase`
- All 10 binaries (9 benchmarks + converter) call `InitSelfRecording(cli.db_dir())`

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

JSON config: specify `graphs`, `benchmarks`, `algorithms`, `trials`, and `options` (symmetrize, verify). See [[Python-Scripts]] for format.

Historical offline-model files may use `results/data/adaptive_models.json`.
The explicit composition path does not require it. Results live under
`results/graphs/`, `results/logs/`, and `results/mappings/`; see
[[Python-Scripts#output-structure]].

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

- [[Contributing]] - Add reordering algorithms
- [[Contributing]] - Add graph algorithms
- [[Python-Scripts]] - Python tools documentation

---

[← Back to Home](Home) | [Python Scripts →](Python-Scripts)
