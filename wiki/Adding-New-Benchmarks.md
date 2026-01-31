# Adding New Benchmarks

This guide explains how to add a new graph algorithm benchmark to GraphBrew.

## Overview

Adding a new benchmark involves:
1. Creating the benchmark source file
2. Implementing the algorithm
3. Integrating with the build system
4. Adding command-line options
5. Testing and validation

---

## Step 1: Create the Source File

### Location

Create your file in `bench/src/`:

```bash
touch bench/src/my_algorithm.cc
```

### Basic Template

```cpp
// Copyright notice
// Description of algorithm

#include <iostream>
#include <algorithm>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "timer.h"

using namespace std;

/*
 * My Algorithm: Brief description
 * 
 * Time Complexity: O(...)
 * Space Complexity: O(...)
 */

typedef float ScoreT;  // Or appropriate type

// Forward declarations
pvector<ScoreT> MyAlgorithm(const Graph &g);
void PrintMyResults(const Graph &g, const pvector<ScoreT> &results);
bool VerifyMyResults(const Graph &g, const pvector<ScoreT> &results);

int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "my_algorithm");
  if (!cli.ParseArgs())
    return -1;
  
  Builder b(cli);
  Graph g = b.MakeGraph();
  
  // BenchmarkKernel takes: cli, graph, kernel_func, print_func, verify_func
  BenchmarkKernel(cli, g, MyAlgorithm, PrintMyResults, VerifyMyResults);
  
  return 0;
}
```

---

## Step 2: Implement the Algorithm

### Basic Structure

```cpp
pvector<ScoreT> MyAlgorithm(const Graph &g) {
  Timer t;
  t.Start();
  
  NodeID num_nodes = g.num_nodes();
  pvector<ScoreT> results(num_nodes);
  
  // Initialize
  #pragma omp parallel for
  for (NodeID n = 0; n < num_nodes; n++) {
    results[n] = 0;
  }
  
  // Main algorithm
  // ...
  
  t.Stop();
  PrintTime("My Algorithm", t.Seconds());
  
  return results;
}
```

### Example: Simple Degree Centrality

```cpp
pvector<float> DegreeCentrality(const Graph &g) {
  Timer t;
  t.Start();
  
  NodeID num_nodes = g.num_nodes();
  pvector<float> centrality(num_nodes);
  
  // Compute normalized degree centrality
  #pragma omp parallel for
  for (NodeID n = 0; n < num_nodes; n++) {
    centrality[n] = static_cast<float>(g.out_degree(n)) / (num_nodes - 1);
  }
  
  t.Stop();
  PrintTime("Degree Centrality", t.Seconds());
  
  return centrality;
}
```

### Example: Iterative Algorithm (like PageRank)

```cpp
pvector<float> MyIterativeAlgorithm(const Graph &g, int max_iters, float tolerance) {
  Timer t;
  t.Start();
  
  NodeID num_nodes = g.num_nodes();
  pvector<float> scores(num_nodes);
  pvector<float> scores_new(num_nodes);
  
  // Initialize
  float init_value = 1.0f / num_nodes;
  #pragma omp parallel for
  for (NodeID n = 0; n < num_nodes; n++) {
    scores[n] = init_value;
  }
  
  // Iterate
  int iter = 0;
  float error = tolerance + 1;
  
  while (iter < max_iters && error > tolerance) {
    error = 0;
    
    #pragma omp parallel for reduction(+:error)
    for (NodeID u = 0; u < num_nodes; u++) {
      float sum = 0;
      for (NodeID v : g.in_neigh(u)) {
        sum += scores[v] / g.out_degree(v);
      }
      scores_new[u] = sum;
      error += fabs(scores_new[u] - scores[u]);
    }
    
    swap(scores, scores_new);
    iter++;
  }
  
  t.Stop();
  PrintTime("My Algorithm (" + to_string(iter) + " iterations)", t.Seconds());
  
  return scores;
}
```

---

## Step 3: Add Verification (Optional but Recommended)

### Purpose

Verification ensures your algorithm produces correct results.

### Example

```cpp
bool VerifyMyResults(const Graph &g, const pvector<ScoreT> &results) {
  NodeID num_nodes = g.num_nodes();
  
  // Check basic properties
  for (NodeID n = 0; n < num_nodes; n++) {
    // Scores should be non-negative
    if (results[n] < 0) {
      cout << "Invalid score at node " << n << ": " << results[n] << endl;
      return false;
    }
  }
  
  // Check sum (if applicable)
  float sum = 0;
  for (NodeID n = 0; n < num_nodes; n++) {
    sum += results[n];
  }
  if (fabs(sum - 1.0f) > 1e-4) {
    cout << "Scores don't sum to 1: " << sum << endl;
    return false;
  }
  
  return true;
}
```

---

## Step 4: Add Output Function

```cpp
void PrintMyResults(const Graph &g, const pvector<ScoreT> &results) {
  NodeID num_nodes = g.num_nodes();
  
  // Find top-k nodes
  const int k = 10;
  vector<pair<ScoreT, NodeID>> top_nodes(num_nodes);
  
  for (NodeID n = 0; n < num_nodes; n++) {
    top_nodes[n] = {results[n], n};
  }
  
  partial_sort(top_nodes.begin(), top_nodes.begin() + k, top_nodes.end(),
               greater<pair<ScoreT, NodeID>>());
  
  cout << "Top " << k << " nodes:" << endl;
  for (int i = 0; i < k && i < num_nodes; i++) {
    cout << "  " << top_nodes[i].second << ": " << top_nodes[i].first << endl;
  }
}
```

---

## Step 5: Add to Build System

### Edit Makefile

Add your binary to the `KERNELS` list in the Makefile:

```makefile
# Current KERNELS list:
KERNELS = bc bfs cc cc_sv pr pr_spmv sssp tc tc_p my_algorithm

# Rule is automatic if your file follows naming convention
# bench/src/my_algorithm.cc -> bench/bin/my_algorithm
```

### Build and Test

```bash
make my_algorithm
./bench/bin/my_algorithm -f scripts/test/graphs/tiny/tiny.el -s -n 3
```

---

## Step 6: Add Command-Line Options

### Using Existing Options

The `CLApp` class provides standard options:
- `-f` : Input file
- `-o` : Ordering algorithm
- `-n` : Number of trials
- `-s` : Symmetrize
- `-r` : Root vertex
- `-v` : Verify

### Adding Custom Options

```cpp
class CLMyAlgorithm : public CLApp {
 public:
  CLMyAlgorithm(int argc, char* argv[], string name) : CLApp(argc, argv, name) {
    get_args_();
  }
  
  int my_param() const { return my_param_; }
  float my_threshold() const { return my_threshold_; }

 private:
  int my_param_ = 10;  // default value
  float my_threshold_ = 0.001;
  
  void get_args_() override {
    CLApp::get_args_();
    my_param_ = cli_.get_int_default("-p", my_param_);
    my_threshold_ = cli_.get_float_default("-t", my_threshold_);
  }
};

// In main:
int main(int argc, char* argv[]) {
  CLMyAlgorithm cli(argc, argv, "my_algorithm");
  // ...
  int param = cli.my_param();
  float threshold = cli.my_threshold();
  // ...
}
```

---

## Complete Example: K-Core Decomposition

### bench/src/kcore.cc

```cpp
// K-Core Decomposition
// Finds the k-core number for each vertex

#include <iostream>
#include <algorithm>
#include <queue>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "timer.h"

using namespace std;

typedef int32_t CoreT;

pvector<CoreT> KCoreDecomposition(const Graph &g) {
  Timer t;
  t.Start();
  
  NodeID num_nodes = g.num_nodes();
  pvector<CoreT> core_num(num_nodes);
  pvector<int32_t> degree(num_nodes);
  
  // Initialize degrees
  #pragma omp parallel for
  for (NodeID n = 0; n < num_nodes; n++) {
    degree[n] = g.out_degree(n);
    core_num[n] = degree[n];
  }
  
  // Find max degree for bucket sort
  int32_t max_deg = 0;
  for (NodeID n = 0; n < num_nodes; n++) {
    max_deg = max(max_deg, degree[n]);
  }
  
  // Bucket-based k-core (Batagelj-Zaversnik algorithm)
  vector<vector<NodeID>> buckets(max_deg + 1);
  pvector<int32_t> bucket_idx(num_nodes);
  
  for (NodeID n = 0; n < num_nodes; n++) {
    bucket_idx[n] = buckets[degree[n]].size();
    buckets[degree[n]].push_back(n);
  }
  
  int32_t curr_core = 0;
  for (int32_t k = 0; k <= max_deg; k++) {
    while (!buckets[k].empty()) {
      NodeID u = buckets[k].back();
      buckets[k].pop_back();
      
      core_num[u] = k;
      
      for (NodeID v : g.out_neigh(u)) {
        if (degree[v] > k) {
          // Remove from current bucket
          int32_t old_deg = degree[v];
          NodeID swap_node = buckets[old_deg].back();
          buckets[old_deg][bucket_idx[v]] = swap_node;
          bucket_idx[swap_node] = bucket_idx[v];
          buckets[old_deg].pop_back();
          
          // Add to lower bucket
          degree[v]--;
          bucket_idx[v] = buckets[degree[v]].size();
          buckets[degree[v]].push_back(v);
        }
      }
    }
  }
  
  t.Stop();
  PrintTime("K-Core Decomposition", t.Seconds());
  
  // Print statistics
  int32_t max_core = 0;
  for (NodeID n = 0; n < num_nodes; n++) {
    max_core = max(max_core, core_num[n]);
  }
  cout << "Maximum core number: " << max_core << endl;
  
  return core_num;
}

bool VerifyKCore(const Graph &g, const pvector<CoreT> &core_num) {
  NodeID num_nodes = g.num_nodes();
  
  // For each vertex, check it has at least k neighbors with core >= k
  for (NodeID u = 0; u < num_nodes; u++) {
    int32_t k = core_num[u];
    int32_t valid_neighbors = 0;
    
    for (NodeID v : g.out_neigh(u)) {
      if (core_num[v] >= k) {
        valid_neighbors++;
      }
    }
    
    if (valid_neighbors < k) {
      cout << "Node " << u << " has core " << k 
           << " but only " << valid_neighbors << " neighbors with core >= " << k << endl;
      return false;
    }
  }
  
  return true;
}

void PrintKCoreResults(const Graph &g, const pvector<CoreT> &core_num) {
  NodeID num_nodes = g.num_nodes();
  
  // Count nodes per core
  int32_t max_core = 0;
  for (NodeID n = 0; n < num_nodes; n++) {
    max_core = max(max_core, core_num[n]);
  }
  
  vector<int64_t> core_counts(max_core + 1, 0);
  for (NodeID n = 0; n < num_nodes; n++) {
    core_counts[core_num[n]]++;
  }
  
  cout << "\nCore distribution:" << endl;
  for (int32_t k = 0; k <= max_core; k++) {
    if (core_counts[k] > 0) {
      cout << "  Core " << k << ": " << core_counts[k] << " nodes" << endl;
    }
  }
}

int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "kcore");
  if (!cli.ParseArgs())
    return -1;
  
  Builder b(cli);
  Graph g = b.MakeGraph();
  
  // BenchmarkKernel takes: cli, graph, kernel_func, print_func, verify_func
  BenchmarkKernel(cli, g, KCoreDecomposition, PrintKCoreResults, VerifyKCore);
  
  return 0;
}
```

### Add to Makefile

```makefile
KERNELS = bc bfs cc cc_sv pr pr_spmv sssp tc tc_p kcore
```

### Test

```bash
make kcore
./bench/bin/kcore -f scripts/test/graphs/tiny/tiny.el -s -n 3 -v
```

---

## Integration with Python Scripts

### Add to lib/utils.py

Edit `scripts/lib/utils.py` to add the benchmark to the BENCHMARKS list:

```python
# Current BENCHMARKS list:
BENCHMARKS = ["pr", "bfs", "cc", "sssp", "bc", "tc", "my_algorithm"]
```

### Use in Experiments

Run experiments with your new benchmark:

```bash
# Include your benchmark in the experiment
python3 scripts/graphbrew_experiment.py \
    --phase benchmark \
    --benchmarks pr bfs kcore \
    --graphs small \
    --trials 5
```

---

## Best Practices

### Performance

1. **Parallelize**: Use OpenMP for loops
   ```cpp
   #pragma omp parallel for
   #pragma omp parallel for reduction(+:sum)
   ```

2. **Cache efficiency**: Access arrays sequentially
   ```cpp
   // Good: sequential access
   for (NodeID n = 0; n < num_nodes; n++) { scores[n] = ...; }
   
   // Bad: random access
   for (NodeID n : random_order) { scores[n] = ...; }
   ```

3. **Minimize allocations**: Reuse buffers
   ```cpp
   // Allocate once, reuse
   pvector<float> temp(num_nodes);
   for (int iter = 0; iter < max_iters; iter++) {
     // use temp
   }
   ```

### Correctness

1. **Add verification**: Catch bugs early
2. **Test on small graphs**: Use test/graphs/
3. **Compare with reference**: Use known implementations

### Documentation

1. **Header comment**: Algorithm description, complexity
2. **Inline comments**: Explain non-obvious logic
3. **Reference paper**: If based on published work

---

## Debugging

### Enable Debug Output

```cpp
#ifdef DEBUG
cout << "Debug: processing node " << n << endl;
#endif
```

Build with:
```bash
make DEBUG=1 my_algorithm
```

### Check for Race Conditions

```cpp
// Bad: race condition
#pragma omp parallel for
for (NodeID u = 0; u < num_nodes; u++) {
  for (NodeID v : g.out_neigh(u)) {
    scores[v] += something;  // Race!
  }
}

// Good: atomic update
#pragma omp parallel for
for (NodeID u = 0; u < num_nodes; u++) {
  for (NodeID v : g.out_neigh(u)) {
    #pragma omp atomic
    scores[v] += something;
  }
}

// Better: thread-local then reduce
```

### Memory Issues

```bash
# Check with valgrind
valgrind --leak-check=full ./bench/bin/my_algorithm -f scripts/test/graphs/tiny/tiny.el -s
```

---

## Next Steps

- [[Adding-New-Algorithms]] - Add reordering algorithms
- [[Code-Architecture]] - Understand the codebase
- [[Running-Benchmarks]] - Test your new benchmark

---

[← Back to Home](Home) | [Code Architecture →](Code-Architecture)
