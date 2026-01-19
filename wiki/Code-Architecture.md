# Code Architecture

Understanding the GraphBrew codebase structure for developers.

## Directory Structure

```
GraphBrew/
├── bench/                    # Core C++ benchmark code
│   ├── bin/                  # Compiled binaries
│   ├── include/              # Header libraries
│   │   ├── corder/           # Corder reordering
│   │   ├── gapbs/            # GAP Benchmark Suite core
│   │   ├── gorder/           # Gorder reordering
│   │   ├── leiden/           # Leiden community detection
│   │   └── rabbit/           # Rabbit Order reordering
│   ├── src/                  # Benchmark source files
│   └── backups/              # Backup files
│
├── scripts/                  # Python tools
│   ├── analysis/             # Analysis scripts
│   ├── brew/                 # Experiment runner
│   ├── config/               # Configuration files
│   ├── test/                 # Test scripts
│   └── perceptron_weights.json
│
├── test/                     # Test files
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

| File | Purpose |
|------|---------|
| [graph.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/gapbs/graph.h) | CSRGraph class, core data structure |
| [builder.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/gapbs/builder.h) | Graph loading and reordering |
| [benchmark.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/gapbs/benchmark.h) | Benchmark harness |
| [command_line.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/gapbs/command_line.h) | CLI parsing |
| [pvector.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/gapbs/pvector.h) | Parallel-friendly vector |
| [timer.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/gapbs/timer.h) | High-resolution timing |

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

```cpp
class Builder {
 public:
  Builder(const CLBase &cli);
  
  // Main entry point
  CSRGraph<NodeID, DestID, WeightT> MakeGraph();
  
  // Reordering
  ReorderingAlgo SquashList(const CSRGraph& g, ReorderingAlgo algo);
  
 private:
  // Graph loading
  EdgeList ReadEdgeList(string filename);
  CSRGraph MakeGraphFromEL(EdgeList& el);
  
  // Reordering implementation
  pvector<NodeID> GenerateMapping(const CSRGraph& g, ReorderingAlgo algo);
  CSRGraph RelabelGraph(const CSRGraph& g, const pvector<NodeID>& mapping);
};
```

---

### Reordering Algorithms

#### Hub-Based (bench/include/gapbs/builder.h)

```cpp
// Degree-Based Grouping
template <typename NodeID>
pvector<NodeID> DBGOrder(const CSRGraph<NodeID>& g) {
  // Groups vertices by log2(degree)
  // Hot vertices (high degree) grouped together
}

// Hub Sorting
template <typename NodeID>  
pvector<NodeID> HubSortOrder(const CSRGraph<NodeID>& g, bool reverse = true) {
  // Sorts by degree, high-degree first
}

// Hub Clustering
template <typename NodeID>
pvector<NodeID> HubClusterOrder(const CSRGraph<NodeID>& g) {
  // Clusters hot vertices together
}
```

#### Community-Based

| Algorithm | File | Method |
|-----------|------|--------|
| RabbitOrder | `rabbit/rabbit_order.hpp` | Community + recursion |
| Gorder | `gorder/GoGraph.h` | Window optimization |
| Corder | `corder/global.h` | Cache-aware ordering |
| RCM | Built-in | Cuthill-McKee |

#### Leiden-Based (bench/include/leiden/)

```cpp
// Community detection
template <class G>
auto leidenCommunities(const G& g, const LeidenOptions& opts);

// Ordering within communities
template <class G>
pvector<NodeID> leidenOrder(const G& g, const Communities& comms);

// Hybrid strategies
template <class G>
pvector<NodeID> leidenDFSOrder(const G& g);
template <class G>
pvector<NodeID> leidenBFSOrder(const G& g);
template <class G>
pvector<NodeID> leidenHybridOrder(const G& g);
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
| `pr.cc` | PageRank | `PageRankPull()` |
| `bfs.cc` | BFS | `BFS_Direction()` |
| `cc.cc` | Connected Components | `Afforest()` |
| `sssp.cc` | Shortest Paths | `DeltaStep()` |
| `bc.cc` | Betweenness | `Brandes()` |
| `tc.cc` | Triangles | `OrderedCount()` |

---

### Python Scripts (scripts/)

#### graph_brew.py - Main Runner

```python
def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run benchmarks
    results = []
    for graph in config['graphs']:
        for algorithm in config['algorithms']:
            for benchmark in config['benchmarks']:
                result = run_benchmark(graph, algorithm, benchmark)
                results.append(result)
    
    # Output results
    save_results(results, args.output)
```

#### analysis/correlation_analysis.py

```python
def compute_perceptron_weights(results_df, weights_file):
    """
    Compute weights from benchmark results.
    
    1. Load existing weights (or use defaults)
    2. For each algorithm with results:
       - Extract features
       - Correlate with performance
       - Update weights
    3. Save merged weights
    """
```

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
  "algorithms": [0, 7, 12, 20],
  "trials": 5,
  "options": {
    "symmetrize": true,
    "verify": false
  }
}
```

### Config Locations

```
scripts/config/
├── brew/           # Standard benchmark configs
│   ├── run.json
│   ├── convert.json
│   └── label.json
├── gap/            # GAP suite compatible
└── test/           # Testing configs
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
./bench/bin/pr -f test/graphs/4.el -s -n 1

# Full test
make test

# Memory check
valgrind ./bench/bin/pr -f test/graphs/4.el -s -n 1
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
