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

[bench/include/gapbs/builder.h](../bench/include/gapbs/builder.h)

### Find the Enum

```cpp
enum ReorderingAlgo {
  ORIGINAL = 0,
  RANDOM = 1,
  SORT = 2,
  HUBSORT = 3,
  HUBCLUSTER = 4,
  DBG = 5,
  HUBSORTDBG = 6,
  HUBCLUSTERDBG = 7,
  RABBITORDER = 8,
  GORDER = 9,
  CORDER = 10,
  RCM = 11,
  LeidenOrder = 12,
  GraphBrewOrder = 13,
  // MAP = 14,  // Reserved
  AdaptiveOrder = 15,
  LeidenDFS = 16,
  LeidenDFSHub = 17,
  LeidenDFSSize = 18,
  LeidenBFS = 19,
  LeidenHybrid = 20,
  // ADD YOUR ALGORITHM HERE
  MY_NEW_ORDER = 21,
};
```

### Naming Convention

- Use CamelCase
- Be descriptive: `LocalitySensitiveOrder`
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

## Step 3: Update the Switch Statement

### Location

In `builder.h`, find the `ReorderGraph` function:

```cpp
template <typename NodeID, typename DestID = NodeID, typename WeightT = NodeID>
CSRGraph<NodeID, DestID, WeightT> ReorderGraph(
    const CSRGraph<NodeID, DestID, WeightT>& g, 
    ReorderingAlgo algo) {
  
  pvector<NodeID> new_ids;
  
  switch (algo) {
    case ORIGINAL:
      return g;  // No reordering
    
    case RANDOM:
      new_ids = RandomOrder(g);
      break;
    
    // ... other cases ...
    
    // ADD YOUR CASE HERE
    case MY_NEW_ORDER:
      new_ids = MyNewReorder(g);
      break;
    
    default:
      cerr << "Unknown reordering algorithm: " << algo << endl;
      exit(1);
  }
  
  return RelabelGraph(g, new_ids);
}
```

---

## Step 4: Add Algorithm Name Mapping

### For Display Output

```cpp
string GetAlgorithmName(ReorderingAlgo algo) {
  switch (algo) {
    case ORIGINAL: return "Original";
    case RANDOM: return "Random";
    // ...
    case MY_NEW_ORDER: return "MyNewOrder";
    default: return "Unknown";
  }
}
```

### For Perceptron (if using AdaptiveOrder)

In the perceptron weights loading:

```cpp
const map<string, ReorderingAlgo> name_to_algo = {
  {"ORIGINAL", ORIGINAL},
  {"RANDOM", RANDOM},
  // ...
  {"MyNewOrder", MY_NEW_ORDER},
};
```

---

## Step 5: Add Perceptron Weights (Optional)

If you want AdaptiveOrder to consider your algorithm:

### Edit scripts/perceptron_weights.json

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
    "w_hub_concentration": 0.0
  }
}
```

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

### 1. Add Enum

```cpp
enum ReorderingAlgo {
  // ...existing...
  LocalitySensitiveOrder = 21,
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

### 5. Test

```bash
make clean && make all
./bench/bin/pr -f test/graphs/4.el -s -o 21 -n 3
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
3. **Test with small graphs first**: Use test/graphs/4.el

### Integration

1. **Consistent naming**: Match enum name, function name, JSON key
2. **Document complexity**: Add comments about time/space complexity
3. **Add tests**: Create test cases for your algorithm

---

## Testing Your Algorithm

### Unit Test

```bash
# Test on small graph
./bench/bin/pr -f test/graphs/4.el -s -o 21 -n 3

# Verify ordering is valid
./bench/bin/pr -f test/graphs/4.el -s -o 21 -n 1 2>&1 | grep -i error
```

### Performance Test

```bash
# Compare with baseline
./bench/bin/pr -f large_graph.el -s -o 0 -n 5   # Baseline
./bench/bin/pr -f large_graph.el -s -o 21 -n 5  # Your algorithm
```

### Memory Test

```bash
# Check for leaks
valgrind ./bench/bin/pr -f test/graphs/4.el -s -o 21 -n 1
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
