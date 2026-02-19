# Adding New Reordering Algorithms

This guide explains how to add a new vertex reordering algorithm to GraphBrew.

## Overview

Adding a new algorithm involves:
1. Adding an enum value
2. Implementing the reordering function
3. Updating the switch statement
4. (Optional) Adding perceptron weights

---

## Step 1: Add Enum Value

### Location

[bench/include/external/gapbs/util.h](https://github.com/UVA-LavaLab/GraphBrew/blob/main/bench/include/external/gapbs/util.h)

### Find the Enum

```cpp
enum ReorderingAlgo {
    ORIGINAL = 0,
    Random = 1,
    Sort = 2,
    HubSort = 3,
    HubCluster = 4,
    DBG = 5,
    HubSortDBG = 6,
    HubClusterDBG = 7,
    RabbitOrder = 8,        // Has variants (see RABBITORDER_VARIANTS in utils.py)
    GOrder = 9,
    COrder = 10,
    RCMOrder = 11,
    GraphBrewOrder = 12,    // Has variants (see GRAPHBREW_VARIANTS in utils.py)
    MAP = 13,               // Load reordering from file
    AdaptiveOrder = 14,     // ML-based perceptron selector
    // Leiden algorithms (15) - grouped together for easier sweeping
    LeidenOrder = 15,       // Format: 15:resolution (GVE-Leiden baseline)
    // LeidenCSR (16) deprecated — GraphBrew (12) subsumes it
    // ADD YOUR ALGORITHM HERE
    MyNewOrder = 16,
};

// Note: Variant lists are defined in scripts/lib/utils.py for Python integration
// (RABBITORDER_VARIANTS, GRAPHBREW_VARIANTS, RCM_VARIANTS)
```

### Naming Convention

- Use PascalCase (e.g., `MyNewOrder`, `LocalitySensitiveOrder`)
- Be descriptive
- Add comment with brief description

---

## Step 2: Implement the Reordering Function

### Function Signature

```cpp
template <typename NodeID>
pvector<NodeID> MyNewReorder(const CSRGraph<NodeID>& g) {
  // Returns mapping: new_id[old_id] = new_vertex_id
  pvector<NodeID> new_ids(g.num_nodes());
  
  // Your algorithm here
  
  return new_ids;
}
```

### Understanding the Return Value

The function returns a permutation where:
- `new_ids[old_vertex_id]` = new vertex ID
- This is used to relabel vertices

### Example: Simple Degree-Based Ordering

```cpp
template <typename NodeID>
pvector<NodeID> DegreeOrder(const CSRGraph<NodeID>& g) {
  NodeID num_nodes = g.num_nodes();
  pvector<NodeID> new_ids(num_nodes);
  
  // Create (degree, vertex) pairs
  vector<pair<int64_t, NodeID>> degree_pairs(num_nodes);
  #pragma omp parallel for
  for (NodeID n = 0; n < num_nodes; n++) {
    degree_pairs[n] = {g.out_degree(n), n};
  }
  
  // Sort by degree (descending)
  sort(degree_pairs.begin(), degree_pairs.end(), greater<pair<int64_t, NodeID>>());
  
  // Assign new IDs
  #pragma omp parallel for
  for (NodeID i = 0; i < num_nodes; i++) {
    new_ids[degree_pairs[i].second] = i;
  }
  
  return new_ids;
}
```

### Example: Using Community Information

```cpp
template <typename NodeID>
pvector<NodeID> CommunityAwareOrder(const CSRGraph<NodeID>& g) {
  NodeID num_nodes = g.num_nodes();
  pvector<NodeID> new_ids(num_nodes);
  
  // Detect communities using Leiden
  auto communities = RunLeidenCommunityDetection(g);
  
  // Group vertices by community, then order within
  NodeID next_id = 0;
  for (const auto& community : communities) {
    for (NodeID v : community.members) {
      new_ids[v] = next_id++;
    }
  }
  
  return new_ids;
}
```

---

## Step 3: Update the Switch Statements

### Location

The main dispatcher is in `bench/include/external/gapbs/builder.h`. The `GenerateMapping` function delegates to standalone implementations in the `reorder/*.h` headers.

#### 1. The `GenerateMapping` Function

Find the `GenerateMapping` function in `builder.h` and add your case:

```cpp
void GenerateMapping(CSRGraph<NodeID_, DestID_, invert> &g,
                     pvector<NodeID_> &new_ids,
                     ReorderingAlgo reordering_algo, bool useOutdeg,
                     std::vector<std::string> reordering_options)
{
    switch (reordering_algo)
    {
    case HubSort:
        GenerateHubSortMapping(g, new_ids, useOutdeg);
        break;
    // ... other cases ...
    
    // ADD YOUR CASE HERE
    case MyNewOrder:
        MyNewReorder(g, new_ids);
        break;
    
    default:
        cerr << "Unknown reordering algorithm: " << reordering_algo << endl;
        exit(1);
    }
}
```

#### 2. The `ReorderingAlgoStr` Function

Add your algorithm name for display:

```cpp
const std::string ReorderingAlgoStr(ReorderingAlgo type)
{
    switch (type)
    {
    // ... existing cases ...
    case MyNewOrder:
        return "MyNewOrder";
    default:
        std::cerr << "Unknown Reordering Algorithm type: " << type << std::endl;
        abort();
    }
}
```

---

## Step 4: Add Algorithm Name Mapping

### For Perceptron Weights (in reorder_types.h)

In `reorder_types.h`, find the `getAlgorithmNameMap()` function and add your algorithm's UPPERCASE base name. This map contains **~16 UPPERCASE base-name entries** with case-insensitive lookup via `lookupAlgorithm()`.

> **Note:** Variant names (e.g., `MyNewOrder_v1`) do NOT go in this map.
> They are auto-discovered by `ParseWeightsFromJSON()` via `VARIANT_PREFIXES`
> prefix scanning, and resolved by `ResolveVariantSelection()` at runtime.

```cpp
inline const std::map<std::string, ReorderingAlgo>& getAlgorithmNameMap() {
    static const std::map<std::string, ReorderingAlgo> name_to_algo = {
        {"ORIGINAL", ORIGINAL}, {"RANDOM", Random}, {"SORT", Sort},
        {"HUBSORT", HubSort}, {"HUBCLUSTER", HubCluster},
        // ... ~16 UPPERCASE base-name entries ...
        {"MYNEWORDER", MyNewOrder},  // ADD YOUR ALGORITHM HERE (UPPERCASE)
    };
    return name_to_algo;
}
```

If your algorithm has variants, add the prefix to the `VARIANT_PREFIXES` array
and add parsing logic in `ResolveVariantSelection()`.

### For Python Scripts

Edit `scripts/lib/utils.py` and add to the `ALGORITHMS` dict:

```python
ALGORITHMS = {
    0: "ORIGINAL",
    # ... existing entries ...
    15: "LeidenOrder",
    16: "MyNewOrder",  # ADD YOUR ALGORITHM HERE
}
```

If your algorithm has variants, add them to the appropriate `*_VARIANTS` tuple
in the **VARIANT REGISTRY** section of `utils.py`. The `get_all_algorithm_variant_names()`
function auto-derives the full canonical name list from the registry — no manual
editing needed.

---

## Step 5: Add Perceptron Weights (Optional)

If you want AdaptiveOrder to consider your algorithm:

### Edit Type Weight Files

When adding a new algorithm, add weights to the type weight files:

```bash
# Files to update (in results/weights/):
results/weights/registry.json                # Contains centroids (no algorithm weights)
results/weights/type_0/weights.json          # Cluster 0 weights
results/weights/type_1/weights.json          # Cluster 1 weights
results/weights/type_N/weights.json          # Additional clusters as needed
```

### Example Entry (add to each type_N.json file)

```json
{
  "MyNewOrder": {
    "bias": 0.5,
    "w_modularity": 0.1,
    "w_log_nodes": 0.0,
    "w_log_edges": 0.0,
    "w_density": 0.0,
    "w_avg_degree": 0.0,
    "w_degree_variance": 0.0,
    "w_hub_concentration": 0.0,
    "w_clustering_coeff": 0.0,
    "w_avg_path_length": 0.0,
    "w_diameter": 0.0,
    "w_community_count": 0.0,
    "w_reorder_time": 0.0,
    "cache_l1_impact": 0.0,
    "cache_l2_impact": 0.0,
    "cache_l3_impact": 0.0,
    "cache_dram_penalty": 0.0,
    "benchmark_weights": {
      "pr": 1.0,
      "bfs": 1.0,
      "cc": 1.0,
      "sssp": 1.0,
      "bc": 1.0,
      "tc": 1.0
    }
  }
}
```

### Automatic Training

After adding the algorithm, run `--train` to train weights from benchmarks:

```bash
python3 scripts/graphbrew_experiment.py --train --size small
```

This will:
1. Benchmark your algorithm on test graphs
2. Compute appropriate weights based on performance
3. Generate per-type weight files automatically

### Weight Guidelines

| Weight | Positive means... |
|--------|-------------------|
| `bias` | Generally preferred (0.3-1.0) |
| `w_modularity` | Better on modular graphs |
| `w_log_nodes` | Better on larger graphs |
| `w_density` | Better on denser graphs |
| `w_hub_concentration` | Better when hubs dominate |

---

## Complete Example: Locality-Sensitive Ordering

### 1. Add Enum (in util.h)

```cpp
enum ReorderingAlgo {
    // ...existing...
    LocalitySensitiveOrder = 16,  // Next available ID after LeidenOrder (15)
};
```

### 2. Implement Algorithm

```cpp
// Locality-sensitive ordering: keeps connected vertices close
template <typename NodeID>
pvector<NodeID> LocalitySensitiveReorder(const CSRGraph<NodeID>& g) {
  NodeID num_nodes = g.num_nodes();
  pvector<NodeID> new_ids(num_nodes, -1);
  pvector<bool> visited(num_nodes, false);
  
  NodeID next_id = 0;
  queue<NodeID> frontier;
  
  // Start from highest degree vertex
  NodeID start = 0;
  int64_t max_deg = g.out_degree(0);
  for (NodeID n = 1; n < num_nodes; n++) {
    if (g.out_degree(n) > max_deg) {
      max_deg = g.out_degree(n);
      start = n;
    }
  }
  
  // BFS traversal for locality
  frontier.push(start);
  visited[start] = true;
  
  while (!frontier.empty() || next_id < num_nodes) {
    if (frontier.empty()) {
      // Find unvisited vertex
      for (NodeID n = 0; n < num_nodes; n++) {
        if (!visited[n]) {
          frontier.push(n);
          visited[n] = true;
          break;
        }
      }
    }
    
    NodeID u = frontier.front();
    frontier.pop();
    new_ids[u] = next_id++;
    
    // Add neighbors in degree order
    vector<pair<int64_t, NodeID>> neighbors;
    for (NodeID v : g.out_neigh(u)) {
      if (!visited[v]) {
        neighbors.push_back({g.out_degree(v), v});
      }
    }
    sort(neighbors.rbegin(), neighbors.rend());
    
    for (auto& [deg, v] : neighbors) {
      if (!visited[v]) {
        visited[v] = true;
        frontier.push(v);
      }
    }
  }
  
  return new_ids;
}
```

### 3. Add to Switch

```cpp
case LocalitySensitiveOrder:
  new_ids = LocalitySensitiveReorder(g);
  break;
```

### 4. Add Weights

```json
{
  "LocalitySensitiveOrder": {
    "bias": 0.6,
    "w_modularity": 0.15,
    "w_log_nodes": 0.05,
    "w_log_edges": 0.05,
    "w_density": -0.1,
    "w_avg_degree": 0.1,
    "w_degree_variance": 0.1,
    "w_hub_concentration": 0.1
  }
}
```

### 5. Add to Python Scripts

Edit `scripts/lib/utils.py`:

```python
ALGORITHMS = {
    # ... existing ...
    16: "LocalitySensitiveOrder",
}
```

### 6. Test

```bash
make clean && make all
# Test your new algorithm (ID 16 = next available after existing 0-15)
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -o 16 -n 3
```

---

## Best Practices

### Performance

1. **Use OpenMP**: Parallelize where possible
   ```cpp
   #pragma omp parallel for
   for (NodeID n = 0; n < num_nodes; n++) { ... }
   ```

2. **Avoid allocations in loops**: Pre-allocate vectors

3. **Cache-friendly access**: Process vertices sequentially when possible

### Correctness

1. **Return valid permutation**: Every vertex must have a new ID
2. **Handle disconnected graphs**: Don't assume connectivity
3. **Test with small graphs first**: Use scripts/test/graphs/tiny/tiny.el

### Integration

1. **Consistent naming**: Match enum name, function name, JSON key
2. **Document complexity**: Add comments about time/space complexity
3. **Add tests**: Create test cases for your algorithm

---

## Testing Your Algorithm

### Unit Test

```bash
# Test on small graph
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -o 16 -n 3

# Verify ordering is valid
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -o 16 -n 1 2>&1 | grep -i error
```

### Performance Test

```bash
# Compare with baseline
./bench/bin/pr -f large_graph.el -s -o 0 -n 5   # Baseline
./bench/bin/pr -f large_graph.el -s -o 16 -n 5  # Your algorithm
```

### Memory Test

```bash
# Check for leaks
valgrind ./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -o 16 -n 1
```

---

## Debugging Tips

### Print Debug Info

```cpp
template <typename NodeID>
pvector<NodeID> MyNewReorder(const CSRGraph<NodeID>& g) {
  #ifdef DEBUG
  cout << "MyNewReorder: num_nodes=" << g.num_nodes() << endl;
  #endif
  // ...
}
```

### Build with Debug

```bash
make clean
make DEBUG=1
```

### Check Permutation Validity

```cpp
bool ValidPermutation(const pvector<NodeID>& perm, NodeID n) {
  vector<bool> seen(n, false);
  for (NodeID i = 0; i < n; i++) {
    if (perm[i] < 0 || perm[i] >= n || seen[perm[i]]) {
      return false;
    }
    seen[perm[i]] = true;
  }
  return true;
}
```

---

## Next Steps

- [[Adding-New-Benchmarks]] - Add new graph algorithms
- [[Code-Architecture]] - Understand the codebase
- [[AdaptiveOrder-ML]] - Train perceptron for your algorithm

---

[← Back to Home](Home) | [Code Architecture →](Code-Architecture)
