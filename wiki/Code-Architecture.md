# Code Architecture

Understanding the GraphBrew codebase structure for developers.

## Directory Structure

```
GraphBrew/
├── bench/                    # Core C++ benchmark code
│   ├── bin/                  # Compiled binaries
│   ├── bin_sim/              # Cache simulation binaries
│   ├── include/              # Header libraries
│   │   ├── cache/            # Cache simulation headers
│   │   ├── corder/           # Corder reordering
│   │   ├── gapbs/            # GAP Benchmark Suite core
│   │   │   └── reorder/      # 📦 Modular reorder headers (~10K lines)
│   │   ├── gorder/           # Gorder reordering
│   │   ├── leiden/           # Leiden community detection
│   │   └── rabbit/           # Rabbit Order reordering
│   ├── src/                  # Benchmark source files
│   ├── src_sim/              # Cache simulation sources
│   └── backups/              # Backup files
│
├── scripts/                  # Python tools (~20,700 lines total)
│   ├── graphbrew_experiment.py  # ⭐ Main orchestration (~3500 lines)
│   ├── perceptron_experiment.py # 🧪 ML weight experimentation
│   ├── adaptive_emulator.py     # 🔍 C++ logic emulation
│   ├── requirements.txt         # Python dependencies
│   │
│   ├── lib/                     # 📦 Core modules (~14,300 lines)
│   │   ├── __init__.py          # Module exports
│   │   ├── types.py             # Data classes (GraphInfo, BenchmarkResult, etc.)
│   │   ├── phases.py            # Phase orchestration
│   │   ├── utils.py             # ALGORITHMS dict, run_command, constants
│   │   ├── features.py          # Graph feature computation
│   │   ├── dependencies.py      # System dependency detection
│   │   ├── download.py          # Graph downloading
│   │   ├── build.py             # Binary compilation
│   │   ├── reorder.py           # Vertex reordering
│   │   ├── benchmark.py         # Benchmark execution
│   │   ├── cache.py             # Cache simulation
│   │   ├── weights.py           # Type-based weight management
│   │   ├── weight_merger.py     # Weight file merging
│   │   ├── training.py          # ML weight training
│   │   ├── analysis.py          # Adaptive analysis
│   │   ├── graph_data.py        # Per-graph data storage
│   │   ├── progress.py          # Progress tracking
│   │   └── results.py           # Result I/O
│   │
│   ├── weights/                 # Auto-generated type weights
│   │   ├── active/              # C++ runtime reads from here
│   │   ├── merged/              # Accumulated from all runs
│   │   ├── runs/                # Historical snapshots
│   │   ├── type_registry.json   # Maps graphs → types + centroids
│   │   └── type_N.json          # Cluster weights
│   │
│   ├── analysis/                # Legacy analysis utilities
│   ├── benchmark/               # Specialized scripts
│   ├── download/                # Standalone downloader
│   └── utils/                   # Additional utilities
│
├── test/                     # (Deprecated) legacy test files; use scripts/test/
│   ├── graphs/               # Sample graphs
│   └── reference/            # Reference outputs
│
├── docs/                     # Documentation
│   └── figures/              # Images
│
└── wiki/                     # This wiki
```

---

## Core Components

### GAP Benchmark Suite (bench/include/gapbs/)

The foundation is built on the GAP Benchmark Suite with extensions.

#### Key Files

| File | Lines | Purpose |
|------|-------|---------|
| [graph.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/gapbs/graph.h) | ~500 | CSRGraph class, core data structure |
| [builder.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/gapbs/builder.h) | ~7,800 | Graph loading and reordering dispatcher |
| [benchmark.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/gapbs/benchmark.h) | ~150 | Benchmark harness |
| [command_line.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/gapbs/command_line.h) | ~300 | CLI parsing |
| [pvector.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/gapbs/pvector.h) | ~150 | Parallel-friendly vector |
| [timer.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/gapbs/timer.h) | ~50 | High-resolution timing |

#### Reorder Module (bench/include/gapbs/reorder/)

The reorder module is a modular header library that extracts reusable components from `builder.h`. It follows an include hierarchy where `reorder_types.h` is the base, and specialized headers extend it.

```
reorder/
├── reorder_types.h      # Base module - types, perceptron, feature computation
├── reorder_adaptive.h   # AdaptiveOrder config and utilities
├── reorder_graphbrew.h  # GraphBrew config and cluster variants
├── reorder_rabbit.h     # RabbitOrder CSR implementation
├── reorder_leiden.h     # GVE-Leiden implementations
├── reorder_hub.h        # Hub-based algorithms (DBG, HubSort, HubCluster)
├── reorder_classic.h    # Classic algorithms (GOrder, COrder, RCM)
└── reorder_basic.h      # Basic algorithms (Original, Random, Sort)
```

| File | Lines | Purpose |
|------|-------|---------|
| `reorder_types.h` | ~3,030 | Common types, perceptron model, selection functions, `ComputeSampledDegreeFeatures` |
| `reorder_adaptive.h` | ~300 | `AdaptiveConfig` struct, adaptive selection utilities |
| `reorder_graphbrew.h` | ~450 | `GraphBrewConfig` struct, `GraphBrewClusterVariant` enum |
| `reorder_leiden.h` | ~3,970 | GVE-Leiden algorithm, dendrogram traversal variants |
| `reorder_rabbit.h` | ~1,040 | RabbitOrder CSR native implementation |
| `reorder_hub.h` | ~600 | Hub-based algorithms (DBG, HubSort, HubCluster, HubClusterDBG) |
| `reorder_classic.h` | ~500 | Classic algorithms (GOrder, COrder, RCM dispatch) |
| `reorder_basic.h` | ~320 | Basic algorithms (Original, Random, Sort) |

**Key Utilities in reorder_types.h:**

```cpp
// Perceptron model for ML-based algorithm selection
struct PerceptronWeights {
    std::map<int, AlgorithmWeights> per_algorithm_weights;
    double ComputeScore(int algo_id, const FeatureVector& features);
};

// Type registry for graph clustering
struct TypeRegistry {
    std::map<std::string, TypeInfo> types;
    std::string FindBestType(const FeatureVector& features);
};

// Sampled degree features for fast topology analysis
struct SampledDegreeFeatures {
    double degree_variance;
    double hub_concentration;
    double avg_degree;
    double clustering_coeff;
};

template<typename GraphT>
SampledDegreeFeatures ComputeSampledDegreeFeatures(const GraphT& g, size_t sample_size = 1000);
```

**Key Configs:**

```cpp
// AdaptiveConfig (reorder_adaptive.h)
struct AdaptiveConfig {
    int max_depth;
    double resolution;
    int min_recurse_size;
    int mode;  // 0 = per-community, 1 = full-graph
    static AdaptiveConfig FromOptions(const std::string& options);
};

// GraphBrewConfig (reorder_graphbrew.h)
struct GraphBrewConfig {
    GraphBrewClusterVariant variant;  // leiden, gve, gveopt, etc.
    int frequency;
    int intra_algo;
    double resolution;
    int maxIterations;
    int maxPasses;
    static GraphBrewConfig FromOptions(const std::string& options);
};
```

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

#### Hub-Based (bench/include/gapbs/builder.h)

```cpp
// Degree-Based Grouping (DBG)
void GenerateDBGMapping(const CSRGraph& g, pvector<NodeID_>& new_ids, bool useOutdeg) {
  // Groups vertices by log2(degree)
  // Hot vertices (high degree) grouped together
}

// Hub Sorting
void GenerateHubSortMapping(const CSRGraph& g, pvector<NodeID_>& new_ids, bool useOutdeg) {
  // Sorts by degree, high-degree first
}

// Hub Clustering
void GenerateHubClusterMapping(const CSRGraph& g, pvector<NodeID_>& new_ids, bool useOutdeg) {
  // Clusters hot vertices together
}

// Hub Cluster with DBG
void GenerateHubClusterDBGMapping(const CSRGraph& g, pvector<NodeID_>& new_ids, bool useOutdeg) {
  // Combines hub clustering with DBG
}
```

#### Community-Based

| Algorithm | File | Method |
|-----------|------|--------|
| RabbitOrder | `rabbit/rabbit_order.hpp` | Community + recursion |
| Gorder | `gorder/GoGraph.h` | Window optimization |
| Corder | `corder/global.h` | Cache-aware ordering |
| RCM | Built-in | Cuthill-McKee |

#### Leiden-Based (bench/include/gapbs/builder.h)

```cpp
// LeidenDendrogram - hierarchical ordering with variants (dfs/dfshub/dfssize/bfs/hybrid)
void GenerateLeidenDendrogramMappingUnified(
    const CSRGraph& g, pvector<NodeID_>& new_ids, 
    const ReorderingOptions& opts);

// LeidenCSR - fast CSR-native ordering with variants
void GenerateLeidenCSRMappingUnified(
    const CSRGraph& g, pvector<NodeID_>& new_ids, 
    const ReorderingOptions& opts);
```

The Leiden community detection algorithm itself is in `bench/include/leiden/leiden.hxx`:

```cpp
// Core Leiden algorithm
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
| `bfs.cc` | BFS | `DOBFS()` |
| `cc.cc` | Connected Components | `Afforest()` |
| `sssp.cc` | Shortest Paths | `DeltaStep()` |
| `bc.cc` | Betweenness | `Brandes()` |
| `tc.cc` | Triangles | `OrderedCount()` |

---

### Python Scripts (scripts/)

The Python tooling follows a modular architecture with a main orchestration script delegating to specialized `lib/` modules.

#### graphbrew_experiment.py - Main Orchestration

The main entry point (~3500 lines) handles argument parsing and delegates to `lib/phases.py`:

```python
def main():
    args = parse_args()
    
    # Discover graphs
    graphs = discover_graphs(args.graphs_dir)
    
    # Create phase configuration
    config = PhaseConfig(
        benchmarks=args.benchmarks,
        trials=args.trials,
        progress=ProgressTracker()
    )
    
    # Run phases via lib/phases.py
    if 'reorder' in args.phases:
        run_reorder_phase(graphs, algorithms, config)
    if 'benchmark' in args.phases:
        run_benchmark_phase(graphs, algorithms, label_maps, config)
    if 'weights' in args.phases:
        run_weights_phase(graphs, results, config)
```

#### lib/phases.py - Phase Orchestration

High-level phase functions for building custom pipelines:

```python
from scripts.lib.phases import (
    PhaseConfig,
    run_reorder_phase,      # Generate vertex reorderings
    run_benchmark_phase,    # Run performance benchmarks
    run_cache_phase,        # Cache simulation
    run_weights_phase,      # Update ML weights
    run_adaptive_phase,     # Adaptive analysis
    run_full_pipeline,      # Complete pipeline
)

config = PhaseConfig(benchmarks=['pr', 'bfs'], trials=3)
results = run_full_pipeline(graphs, algorithms, config)
```

#### lib/types.py - Central Data Classes

All data classes are centralized in `types.py`:

```python
from scripts.lib.types import (
    GraphInfo,        # Graph metadata (name, path, size, nodes, edges)
    BenchmarkResult,  # Benchmark execution result
    CacheResult,      # Cache simulation result
    ReorderResult,    # Reordering result
    PerceptronWeight, # Perceptron weight dataclass
)
```

#### lib/utils.py - Core Utilities

```python
from scripts.lib.utils import (
    ALGORITHMS,      # {0: "ORIGINAL", 1: "RANDOM", 7: "HUBCLUSTERDBG", ...}
    BENCHMARKS,      # ['pr', 'bfs', 'cc', 'sssp', 'bc', 'tc']
    run_command,     # Execute shell commands with timeout
    get_timestamp,   # Formatted timestamps
)
```

#### Module Overview

| Module | Lines | Purpose |
|--------|-------|---------|
| `weights.py` | ~1290 | Weight management |
| `graph_data.py` | ~1220 | Per-graph data storage |
| `download.py` | ~1130 | Graph downloading |
| `phases.py` | ~1080 | Phase orchestration |
| `reorder.py` | ~1070 | Vertex reordering |
| `dependencies.py` | ~890 | Dependency management |
| `analysis.py` | ~880 | Adaptive analysis |
| `weight_merger.py` | ~830 | Weight merging |
| `results.py` | ~750 | Result I/O |
| `utils.py` | ~730 | Core utilities |
| `training.py` | ~720 | ML training |
| `cache.py` | ~630 | Cache simulation |
| `progress.py` | ~600 | Progress tracking |
| `types.py` | ~595 | Data classes |
| `benchmark.py` | ~560 | Benchmark execution |
| `features.py` | ~555 | Graph features |
| `build.py` | ~340 | Binary compilation |

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

### pvector<T>

Parallel-friendly vector with aligned memory:

```cpp
template <typename T>
class pvector {
 public:
  pvector(size_t n);
  pvector(size_t n, T init_val);
  
  T& operator[](size_t n);
  size_t size() const;
  T* data();
  
  // Iterators
  iterator begin();
  iterator end();
};
```

### SlidingQueue<T>

Lock-free queue for BFS:

```cpp
template <typename T>
class SlidingQueue {
 public:
  void push_back(T to_add);
  void slide_window();
  
  // Thread-local buffers for parallel push
  QueueBuffer<T> queue_buf;
};
```

### Bitmap

Efficient bit vector:

```cpp
class Bitmap {
 public:
  Bitmap(size_t size);
  
  void set_bit(size_t pos);
  void set_bit_atomic(size_t pos);
  bool get_bit(size_t pos) const;
  void reset();
};
```

---

## Parallelization Strategy

### OpenMP Usage

```cpp
// Parallel initialization
#pragma omp parallel for
for (NodeID n = 0; n < num_nodes; n++) {
  scores[n] = init_value;
}

// Parallel with reduction
float sum = 0;
#pragma omp parallel for reduction(+:sum)
for (NodeID n = 0; n < num_nodes; n++) {
  sum += scores[n];
}

// Parallel with atomic
#pragma omp parallel for
for (NodeID u = 0; u < num_nodes; u++) {
  for (NodeID v : g.out_neigh(u)) {
    #pragma omp atomic
    counts[v]++;
  }
}
```

### Thread-Local Buffers

For reducing contention:

```cpp
#pragma omp parallel
{
  // Thread-local buffer
  vector<NodeID> local_frontier;
  
  #pragma omp for
  for (NodeID u : current_frontier) {
    for (NodeID v : g.out_neigh(u)) {
      if (try_visit(v)) {
        local_frontier.push_back(v);
      }
    }
  }
  
  // Merge into global
  #pragma omp critical
  next_frontier.insert(next_frontier.end(), 
                       local_frontier.begin(), 
                       local_frontier.end());
}
```

---

## Memory Layout

### CSR (Compressed Sparse Row)

```
Nodes:     [0] [1] [2] [3]
           ↓   ↓   ↓   ↓
Index:     [0] [2] [4] [6] [8]  ← Offsets into neighbors
                                
Neighbors: [1,2|0,2|0,1,3|2]    ← Concatenated adjacency lists
           0's  1's  2's   3's
```

### Memory Access Pattern

```cpp
// Good: Sequential access (cache-friendly)
for (NodeID n = 0; n < num_nodes; n++) {
  for (NodeID v : g.out_neigh(n)) {
    // Neighbors are contiguous in memory
  }
}

// After reordering: frequently accessed nodes are close
// Nodes 0,1,2,3 might become 1000,1001,1002,1003
// if they're in the same community
```

---

## Configuration System

### JSON Config Structure

```json
{
  "graphs": ["facebook.el", "twitter.el"],
  "benchmarks": ["pr", "bfs", "cc"],
  "algorithms": [0, 7, 12, 17],
  "trials": 5,
  "options": {
    "symmetrize": true,
    "verify": false
  }
}
```

### Data Locations

```
scripts/
├── graphbrew_experiment.py    # Main orchestration script
├── weights/                   # Perceptron weight files
│   ├── active/                # C++ runtime reads from here (type_0.json, etc.)
│   │   ├── type_registry.json # Maps graphs → types + algorithm list
│   │   └── type_N.json        # Per-cluster weights
│   ├── merged/                # Accumulated from all runs
│   └── runs/                  # Historical snapshots
└── lib/                       # Python library modules
    ├── utils.py               # ALGORITHMS dict, constants
    └── weights.py             # Weight management and computation

results/
├── graphs/{name}/features.json # Per-graph static features
├── logs/{name}/runs/          # Timestamped experiment data
└── graph_properties_cache.json # Cached properties for type detection
```

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
