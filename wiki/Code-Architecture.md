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
│   │   │   ├── reorder/      # Reordering algorithms (~12,435 lines)
│   │   │   └── partition/    # Partitioning (trust.h, cagra/popt.h)
│   │   ├── external/         # External libraries (bundled)
│   │   │   ├── gapbs/        # Core GAPBS runtime (builder.h ~3,747 lines)
│   │   │   ├── rabbit/       # RabbitOrder
│   │   │   ├── gorder/       # GOrder
│   │   │   ├── corder/       # COrder
│   │   │   └── leiden/       # GVE-Leiden
│   │   └── cache_sim/        # Cache simulation headers
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
│   ├── analysis/                # Analysis utilities
│   ├── benchmark/               # Specialized scripts
│   ├── download/                # Standalone downloader
│   └── utils/                   # Additional utilities
│
│
├── scripts/test/             # Pytest suite
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
| [graph.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/graph.h) | ~500 | CSRGraph class, core data structure |
| [builder.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/builder.h) | ~3,747 | Graph loading and reordering dispatcher |
| [benchmark.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/benchmark.h) | ~150 | Benchmark harness |
| [command_line.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/command_line.h) | ~300 | CLI parsing |
| [pvector.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/pvector.h) | ~150 | Parallel-friendly vector |
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
├── reorder_types.h      # Base: types, perceptron, feature computation (~3,892 lines)
├── reorder_basic.h      # Original, Random, Sort (algo 0-2) (~324 lines)
├── reorder_hub.h        # HubSort, HubCluster, DBG variants (algo 3-7) (~598 lines)
├── reorder_rabbit.h     # RabbitOrder native CSR (algo 8) (~1,161 lines)
├── reorder_classic.h    # GOrder, COrder, RCMOrder (algo 9-11) (~502 lines)
├── reorder_graphbrew.h  # GraphBrew multi-level (algo 12) (~869 lines)
├── reorder_adaptive.h   # ML-based selection (algo 14) (~638 lines)
├── reorder_leiden.h     # Leiden community detection (algo 15-17) (~3,970 lines)
└── reorder.h            # Main dispatcher (~481 lines)
```

**Total: ~12,435 lines**

| File | Lines | Purpose |
|------|-------|---------|
| `reorder_types.h` | ~3,892 | Common types, perceptron model, `EdgeList`, threshold functions |
| `reorder_leiden.h` | ~3,970 | GVE-Leiden algorithm, dendrogram traversal variants |
| `reorder_rabbit.h` | ~1,161 | RabbitOrder CSR native implementation |
| `reorder_graphbrew.h` | ~869 | `GraphBrewConfig`, cluster variants, multi-level reordering |
| `reorder_adaptive.h` | ~638 | `AdaptiveConfig`, ML-based per-community algorithm selection |
| `reorder_hub.h` | ~598 | Hub-based algorithms (DBG, HubSort, HubCluster) |
| `reorder_classic.h` | ~502 | Classic algorithms (GOrder, COrder, RCM dispatch) |
| `reorder.h` | ~481 | Main dispatcher, `ApplyBasicReorderingStandalone` |
| `reorder_basic.h` | ~324 | Basic algorithms (Original, Random, Sort) |

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

**Leiden Algorithm Constants (reorder_leiden.h):**

All Leiden-related functions use these centralized constants for consistency. Function default arguments also use these constants instead of hardcoded values:

```cpp
namespace graphbrew {
namespace leiden {
// Default parameters - Single Source of Truth for C++ Leiden
constexpr double DEFAULT_RESOLUTION = 0.75;
constexpr double DEFAULT_TOLERANCE = 1e-2;
constexpr double DEFAULT_AGGREGATION_TOLERANCE = 0.8;
constexpr double DEFAULT_QUALITY_FACTOR = 10.0;
constexpr int DEFAULT_MAX_ITERATIONS = 10;
constexpr int DEFAULT_MAX_PASSES = 10;

// Variant-specific parameters
constexpr int FAST_MAX_ITERATIONS = 5;
constexpr int FAST_MAX_PASSES = 5;
constexpr int MODULARITY_MAX_ITERATIONS = 20;
constexpr int MODULARITY_MAX_PASSES = 20;
constexpr double SORT_AGGREGATION_TOLERANCE = 0.95;  // Allows more passes
} // namespace leiden
} // namespace graphbrew

// After namespace close, bring constants into global scope for backward compatibility
using graphbrew::leiden::DEFAULT_TOLERANCE;
using graphbrew::leiden::DEFAULT_AGGREGATION_TOLERANCE;
using graphbrew::leiden::DEFAULT_QUALITY_FACTOR;
// ... (all except DEFAULT_RESOLUTION which conflicts with adaptive)

// Function signatures use constants for default arguments:
template <typename K, typename W, typename NodeID_T, typename DestID_T>
GVELeidenResult<K> GVELeidenCSR(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    double resolution = 1.0,
    double tolerance = DEFAULT_TOLERANCE,
    double aggregation_tolerance = DEFAULT_AGGREGATION_TOLERANCE,
    double tolerance_drop = DEFAULT_QUALITY_FACTOR,
    int max_iterations = 20,
    int max_passes = DEFAULT_MAX_PASSES);
```

**Adaptive Algorithm Constants (reorder_adaptive.h):**

```cpp
namespace adaptive {
constexpr int DEFAULT_MODE = 1;           // Per-community
constexpr int DEFAULT_RECURSION_DEPTH = 0;
constexpr double DEFAULT_RESOLUTION = 1.0;  // Different from leiden::DEFAULT_RESOLUTION
constexpr size_t DEFAULT_MIN_RECURSE_SIZE = 50000;
} // namespace adaptive
```

> ⚠️ **Important**: Use `graphbrew::leiden::DEFAULT_RESOLUTION` or `adaptive::DEFAULT_RESOLUTION` explicitly to avoid ambiguity.

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

```cpp
// Degree-Based Grouping (DBG) - standalone template function
template<typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateDBGMappingStandalone(const CSRGraph<NodeID_, DestID_, invert>& g,
                                   pvector<NodeID_>& new_ids, bool useOutdeg) {
  // Groups vertices by log2(degree)
  // Hot vertices (high degree) grouped together
}

// Hub Sorting
template<typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateHubSortMappingStandalone(const CSRGraph<NodeID_, DestID_, invert>& g,
                                       pvector<NodeID_>& new_ids, bool useOutdeg) {
  // Sorts by degree, high-degree first
}

// Hub Clustering
template<typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateHubClusterMappingStandalone(const CSRGraph<NodeID_, DestID_, invert>& g,
                                          pvector<NodeID_>& new_ids, bool useOutdeg) {
  // Clusters hot vertices together
}

// Hub Cluster with DBG
template<typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateHubClusterDBGMappingStandalone(const CSRGraph<NodeID_, DestID_, invert>& g,
                                             pvector<NodeID_>& new_ids, bool useOutdeg) {
  // Combines hub clustering with DBG
}
```

**Edge Case Guards:**

All reordering functions include guards for empty graphs to prevent division-by-zero (FPE) when computing average degree:

```cpp
// In each function (GenerateHubSort, GenerateDBG, COrder, GVELeiden, etc.):
const int64_t num_nodes = g.num_nodes();
const int64_t num_edges = g.num_edges();

// GUARD: Empty graph - nothing to do
if (num_nodes == 0) {
    t.Stop();
    PrintTime("Algorithm Map Time", t.Seconds());
    return;
}

const int64_t avgDegree = num_edges / num_nodes;  // Safe now
```

**Files with FPE guards:**
- `reorder_hub.h` - All 5 hub-based functions
- `reorder_classic.h` - COrder and COrder_v2
- `reorder_graphbrew.h` - GraphBrewHubCluster
- `reorder_leiden.h` - GVELeidenAdaptiveCSR
- `reorder_adaptive.h` - Adaptive algorithm selection
- `reorder.h` - ReorderCommunitySubgraphStandalone

This is important for GraphBrewOrder which may create empty subgraphs for communities with no internal edges (e.g., on Kronecker graphs).

#### Community-Based

| Algorithm | File | Method |
|-----------|------|--------|
| RabbitOrder | `graphbrew/reorder/reorder_rabbit.h` | Native CSR community detection + recursion |
| Gorder | `external/gorder/GoGraph.h` | Window optimization |
| Corder | `external/corder/global.h` | Cache-aware ordering |
| RCM | `graphbrew/reorder/reorder_classic.h` | Cuthill-McKee dispatch |

#### Leiden-Based (bench/include/graphbrew/reorder/reorder_leiden.h)

```cpp
// LeidenDendrogram - hierarchical ordering with variants (dfs/dfshub/dfssize/bfs/hybrid)
template<typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateLeidenDendrogramMappingUnified(
    const CSRGraph<NodeID_, DestID_, invert>& g, pvector<NodeID_>& new_ids,
    const ReorderingOptions& opts);

// LeidenCSR - fast CSR-native ordering with GVE-Leiden variants
template<typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateLeidenCSRMappingUnified(
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

**Single Source of Truth** for all algorithm/variant/size/timeout constants:

```python
from scripts.lib.utils import (
    # Algorithm definitions
    ALGORITHMS,      # {0: "ORIGINAL", 1: "RANDOM", ..., 17: "LeidenCSR"}
    ALGORITHM_IDS,   # Reverse: {"ORIGINAL": 0, ...}
    SLOW_ALGORITHMS, # {9, 10, 11} - Gorder, Corder, RCM
    BENCHMARKS,      # ['pr', 'bfs', 'cc', 'sssp', 'bc', 'tc']
    
    # Variant lists (authoritative definitions)
    LEIDEN_CSR_VARIANTS,        # ['gve', 'gveopt', 'gveopt2', ...]
    GRAPHBREW_VARIANTS,         # ['leiden', 'gve', 'gveopt', ...]
    RABBITORDER_VARIANTS,       # ['csr', 'boost']
    LEIDEN_DENDROGRAM_VARIANTS, # ['dfs', 'dfshub', 'dfssize', 'bfs', 'hybrid']
    
    # Graph size thresholds (MB)
    SIZE_SMALL,      # 50 MB
    SIZE_MEDIUM,     # 500 MB
    SIZE_LARGE,      # 2000 MB
    SIZE_XLARGE,     # 10000 MB
    
    # Timeout constants (seconds)
    TIMEOUT_REORDER,     # 43200 (12 hours)
    TIMEOUT_BENCHMARK,   # 600 (10 min)
    TIMEOUT_SIM,         # 1200 (20 min)
    TIMEOUT_SIM_HEAVY,   # 3600 (1 hour)
    
    # Utilities
    run_command,     # Execute shell commands with timeout
    get_timestamp,   # Formatted timestamps
)
```

> ⚠️ **Important**: All constants (algorithms, variants, sizes, benchmarks, timeouts) are defined ONLY in `utils.py`. Other modules import from here - never duplicate these definitions.

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
