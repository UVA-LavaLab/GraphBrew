# Cache Simulation

The Cache Simulation framework in GraphBrew provides detailed cache performance analysis for graph algorithms. This allows you to understand memory access patterns and cache efficiency at L1, L2, and L3 levels.

## Overview

Graph algorithms are memory-intensive and cache performance significantly impacts execution time. The cache simulation framework instruments graph algorithms to track every memory access, simulating how data flows through a three-level cache hierarchy.

**Key Features:**
- Multi-level cache hierarchy (L1/L2/L3)
- Configurable cache parameters (size, associativity, line size)
- Six eviction policies (LRU, FIFO, RANDOM, LFU, PLRU, SRRIP)
- Detailed per-level statistics
- JSON export for analysis integration
- Feature vector output for perceptron training

## Quick Start

```bash
# Build all cache simulation binaries
make all-sim

# Run PageRank with cache simulation
./bench/bin_sim/pr -g 15 -n 1

# Run with custom cache configuration
CACHE_L1_SIZE=65536 CACHE_POLICY=LFU ./bench/bin_sim/bfs -g 15 -n 1
```

## Architecture

```
bench/
├── include/cache/
│   ├── cache_sim.h     # Cache simulator core
│   └── graph_sim.h     # Graph wrappers for tracking
├── src_sim/            # Instrumented algorithms
│   ├── pr.cc           # PageRank
│   ├── bfs.cc          # Breadth-First Search
│   ├── bc.cc           # Betweenness Centrality
│   ├── cc.cc           # Connected Components
│   ├── sssp.cc         # Single-Source Shortest Paths
│   └── tc.cc           # Triangle Counting
└── bin_sim/            # Compiled binaries
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_L1_SIZE` | 32768 | L1 cache size (bytes) |
| `CACHE_L1_WAYS` | 8 | L1 associativity |
| `CACHE_L2_SIZE` | 262144 | L2 cache size (bytes) |
| `CACHE_L2_WAYS` | 4 | L2 associativity |
| `CACHE_L3_SIZE` | 8388608 | L3 cache size (bytes) |
| `CACHE_L3_WAYS` | 16 | L3 associativity |
| `CACHE_LINE_SIZE` | 64 | Line size (bytes) |
| `CACHE_POLICY` | LRU | Eviction policy |
| `CACHE_OUTPUT_JSON` | - | JSON output file path |

### Default Cache Configuration

The default configuration models a typical modern CPU:

| Level | Size | Associativity | Lines | Sets |
|-------|------|---------------|-------|------|
| L1 | 32 KB | 8-way | 512 | 64 |
| L2 | 256 KB | 4-way | 4096 | 1024 |
| L3 | 8 MB | 16-way | 131072 | 8192 |

### Eviction Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| **LRU** | Evicts least recently used line | General purpose, best for temporal locality |
| **FIFO** | Evicts oldest line | Simple, deterministic behavior |
| **RANDOM** | Random eviction | Baseline comparison |
| **LFU** | Evicts least frequently used | Good for hot data |
| **PLRU** | Tree-based pseudo-LRU | Hardware-efficient LRU approximation |
| **SRRIP** | Re-reference interval prediction | Scan-resistant, good for streaming |

## Output Format

### Console Statistics

```
================================================================================
                           CACHE SIMULATION RESULTS                             
================================================================================
Configuration:
  L1 Cache: 32768 bytes, 8-way, 64-byte lines (64 sets)
  L2 Cache: 262144 bytes, 4-way, 64-byte lines (1024 sets)
  L3 Cache: 8388608 bytes, 16-way, 64-byte lines (8192 sets)
  Eviction Policy: LRU

Statistics by Level:
--------------------------------------------------------------------------------
  Level      Accesses         Hits       Misses    Hit Rate   Miss Rate
--------------------------------------------------------------------------------
     L1     123456789     98765432     24691357      80.00%      20.00%
     L2      24691357     19753086      4938271      80.00%      20.00%
     L3       4938271      4444444       493827      90.00%      10.00%
--------------------------------------------------------------------------------

Overall Statistics:
  Total Memory Accesses: 123456789
  Total Cache Hits:      122962962 (99.60% of memory)
  Memory Accesses (L3 Misses): 493827 (0.40%)
================================================================================
```

### JSON Export

```bash
CACHE_OUTPUT_JSON=results.json ./bench/bin_sim/pr -g 15 -n 1
```

Output format:
```json
{
  "config": {
    "l1_size": 32768,
    "l1_ways": 8,
    "l2_size": 262144,
    "l2_ways": 4,
    "l3_size": 8388608,
    "l3_ways": 16,
    "line_size": 64,
    "policy": "LRU"
  },
  "levels": {
    "L1": {"accesses": 123456789, "hits": 98765432, "misses": 24691357, "hit_rate": 0.80},
    "L2": {"accesses": 24691357, "hits": 19753086, "misses": 4938271, "hit_rate": 0.80},
    "L3": {"accesses": 4938271, "hits": 4444444, "misses": 493827, "hit_rate": 0.90}
  },
  "feature_vector": [0.80, 0.20, 0.80, 0.20, 0.90, 0.10, 0.996, 0.004]
}
```

## Use Cases

### 1. Comparing Reordering Algorithms

Compare cache efficiency across different reorderings:

```bash
# Test ORIGINAL ordering
./bench/bin_sim/pr -f graph.mtx -o 0 -n 1

# Test RabbitOrder
./bench/bin_sim/pr -f graph.mtx -o 8 -n 1

# Test GraphBrewOrder
./bench/bin_sim/pr -f graph.mtx -o 12:10:17 -n 1
```

### 2. Analyzing Cache Behavior

Study how cache sizes affect performance:

```bash
for size in 16384 32768 65536 131072; do
    CACHE_L1_SIZE=$size ./bench/bin_sim/bfs -g 18 -n 1
done
```

### 3. Perceptron Training Features

The cache simulator provides features for the ML-based algorithm selector. The unified experiment script automatically collects these:

```bash
# Run cache simulation as part of the full pipeline
python3 scripts/graphbrew_experiment.py --full --download-size SMALL

# Or just the cache phase
python3 scripts/graphbrew_experiment.py --phase cache
```

Cache features are stored in `results/cache_*.json` and integrated into perceptron weights:

```json
{
  "LeidenHybrid": {
    "bias": 0.85,
    "w_modularity": 0.25,
    "cache_l1_impact": 0.1,
    "cache_l2_impact": 0.05,
    "cache_l3_impact": 0.02,
    "cache_dram_penalty": -0.1,
    "_metadata": {
      "avg_l1_hit_rate": 85.2,
      "avg_l2_hit_rate": 92.1,
      "avg_l3_hit_rate": 98.5
    }
  }
}
```

The perceptron uses these weights to factor cache performance into algorithm selection.

### 4. Automated Cache Benchmark Suite

Run comprehensive cache benchmarks using the unified script:

```bash
# Run cache simulation on all graphs
python3 scripts/graphbrew_experiment.py --phase cache

# Skip heavy simulations (BC, SSSP) on large graphs
python3 scripts/graphbrew_experiment.py --phase cache --skip-heavy

# Full pipeline includes cache simulation automatically
python3 scripts/graphbrew_experiment.py --full --download-size MEDIUM
```

Results are saved to `results/cache_*.json`:

```json
{
  "graph": "email-Enron",
  "algorithm_id": 20,
  "algorithm_name": "LeidenHybrid",
  "benchmark": "pr",
  "l1_hit_rate": 85.2,
  "l2_hit_rate": 92.1,
  "l3_hit_rate": 98.5,
  "success": true,
  "error": ""
}
```
    --algorithms pr bfs cc bc sssp tc \
    --reorders 0 7 8 12 17 20 \
    --graphs-dir ./graphs \
    --output ./cache_results

# The benchmark generates:
# - cache_results_<timestamp>.json  # Raw cache statistics
# - cache_correlations_<timestamp>.json  # Performance correlations
# - cache_features_<timestamp>.json  # Perceptron training features
```

### 5. Algorithm Selection

Cache characteristics help predict optimal algorithms:

| Cache Pattern | Recommended Algorithm |
|---------------|----------------------|
| High L1 miss, low L3 miss | Random walks tolerable, BFS-based works |
| High L3 miss rate | Needs locality-focused reordering |
| Uniform access pattern | Simple reordering sufficient |
| Skewed hub access | Hub-clustering beneficial |

## Instrumented Algorithms

### PageRank (`pr.cc`)
- Tracks: contribution array, score array, CSR index reads
- Key insight: Pull-based PR has predictable access to outgoing edges

### BFS (`bfs.cc`)
- Tracks: parent array, frontier queue, bitmap, CSR traversal
- Key insight: Direction-optimizing switch point affects cache behavior

### Betweenness Centrality (`bc.cc`)
- Tracks: path counts, depths, deltas, successor bitmaps
- Key insight: Forward/backward passes have different access patterns

### Connected Components (`cc.cc`)
- Tracks: component array, link/compress operations
- Key insight: Sampling phase vs. linking phase cache differences

### SSSP (`sssp.cc`)
- Tracks: distance array, delta-step buckets, weighted neighbors
- Key insight: Delta parameter affects bucket access patterns

### Triangle Counting (`tc.cc`)
- Tracks: neighbor list traversals, intersection operations
- Key insight: Sorted adjacency lists improve intersection cache efficiency

## Limitations

1. **Performance**: ~10-100x slower than uninstrumented code
2. **Threading**: Accesses are serialized for accurate counting
3. **Simplifications**: No cache coherence, prefetching, or speculation modeling
4. **Scope**: Data cache only (no instruction cache or TLB)

## Perceptron Integration

The cache simulation features integrate with the perceptron-based algorithm selector:

### Feature Vector

The perceptron uses 14 features (7 structural + 7 cache):

**Structural Features:**
| Feature | Description |
|---------|-------------|
| `modularity` | Community structure strength |
| `log_nodes` | log₁₀(node count) |
| `log_edges` | log₁₀(edge count) |
| `density` | Edge density |
| `avg_degree` | Average vertex degree |
| `degree_variance` | Degree distribution heterogeneity |
| `hub_concentration` | Power-law hub concentration |

**Cache Features:**
| Feature | Description |
|---------|-------------|
| `l1_hit_rate` | L1 cache hit rate (0.0-1.0) |
| `l2_hit_rate` | L2 cache hit rate (0.0-1.0) |
| `l3_hit_rate` | L3 cache hit rate (0.0-1.0) |
| `dram_access_rate` | Rate of accesses reaching memory |
| `l1_eviction_rate` | L1 cache eviction rate |
| `l2_eviction_rate` | L2 cache eviction rate |
| `l3_eviction_rate` | L3 cache eviction rate |

### Training Pipeline

```bash
# 1. Run full pipeline (includes cache simulation)
python3 scripts/graphbrew_experiment.py --full --download-size SMALL

# 2. Or run cache phase separately
python3 scripts/graphbrew_experiment.py --phase cache --graphs small

# 3. Generate weights with cache features
python3 scripts/graphbrew_experiment.py --phase weights

# 4. Use in benchmarks with AdaptiveOrder
./bench/bin/pr -g 18 -o 14  # AdaptiveOrder uses trained weights
```

Cache features are automatically integrated into perceptron weights:
- `cache_l1_impact`: Bonus for high L1 hit rate
- `cache_l2_impact`: Bonus for high L2 hit rate  
- `cache_l3_impact`: Bonus for high L3 hit rate
- `cache_dram_penalty`: Penalty for DRAM accesses

### C++ Integration

The `getFeatures()` method in `cache_sim.h` returns the cache feature vector:

```cpp
CacheSimulator cache;
// ... run simulation ...

std::vector<double> features = cache.getFeatures();
// [l1_hit_rate, l2_hit_rate, l3_hit_rate, dram_access_rate,
//  l1_eviction_rate, l2_eviction_rate, l3_eviction_rate]
```

## Related Pages

- [[Perceptron-Weights]] - Using cache features for algorithm selection
- [[Correlation-Analysis]] - Correlating cache stats with performance
- [[Benchmark-Suite]] - Running performance experiments

---

[← Back to Home](Home)
