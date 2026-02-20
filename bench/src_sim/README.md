# Cache Simulation Framework

This directory contains cache-instrumented versions of GraphBrew's graph algorithms. Each algorithm has been modified to track memory accesses through a configurable cache simulator, allowing analysis of cache performance (hits/misses) at L1, L2, and L3 levels.

## Overview

The cache simulation framework provides:
- **Multi-level cache hierarchy**: Simulates L1, L2, and L3 caches
- **Configurable parameters**: Cache sizes, associativity, line size
- **Multiple eviction policies**: LRU, FIFO, RANDOM, LFU, PLRU, SRRIP
- **Detailed statistics**: Hits, misses, hit rate, miss rate per level
- **JSON export**: For integration with analysis tools and perceptron features

## Building

```bash
# Build all simulation binaries
make all-sim

# Build specific algorithm
make sim-pr
make sim-bfs
make sim-bc
make sim-cc
make sim-sssp
make sim-tc
```

## Running

```bash
# Run with default cache configuration
./bench/bin_sim/pr -g 15 -n 1

# Run with custom cache configuration via environment variables
CACHE_L1_SIZE=32768 CACHE_L1_WAYS=8 CACHE_POLICY=LRU ./bench/bin_sim/pr -g 15 -n 1

# Export statistics to JSON
CACHE_OUTPUT_JSON=cache_stats.json ./bench/bin_sim/pr -g 15 -n 1
```

## Configuration

All cache parameters are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_L1_SIZE` | 32768 (32KB) | L1 cache size in bytes |
| `CACHE_L1_WAYS` | 8 | L1 set associativity |
| `CACHE_L2_SIZE` | 262144 (256KB) | L2 cache size in bytes |
| `CACHE_L2_WAYS` | 4 | L2 set associativity |
| `CACHE_L3_SIZE` | 8388608 (8MB) | L3 cache size in bytes |
| `CACHE_L3_WAYS` | 16 | L3 set associativity |
| `CACHE_LINE_SIZE` | 64 | Cache line size in bytes |
| `CACHE_POLICY` | LRU | Eviction policy |

### Eviction Policies

- **LRU** (Least Recently Used): Evicts the cache line that hasn't been accessed for the longest time
- **FIFO** (First In First Out): Evicts the oldest cache line
- **RANDOM**: Randomly selects a line to evict
- **LFU** (Least Frequently Used): Evicts the cache line with the fewest accesses
- **PLRU** (Pseudo-LRU): Tree-based approximation of LRU
- **SRRIP** (Static Re-Reference Interval Prediction): Uses re-reference prediction values

## Output

### Console Output

After algorithm execution, cache statistics are printed:

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

### JSON Output

When `CACHE_OUTPUT_JSON` is set, statistics are exported in JSON format:

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
    "L1": { "accesses": 123456789, "hits": 98765432, "misses": 24691357, "hit_rate": 0.80 },
    "L2": { "accesses": 24691357, "hits": 19753086, "misses": 4938271, "hit_rate": 0.80 },
    "L3": { "accesses": 4938271, "hits": 4444444, "misses": 493827, "hit_rate": 0.90 }
  },
  "feature_vector": [0.80, 0.20, 0.80, 0.20, 0.90, 0.10, 0.996, 0.004]
}
```

## Instrumented Algorithms

| File | Algorithm | Description |
|------|-----------|-------------|
| `pr.cc` | PageRank | Pull-based PageRank with tolerance convergence |
| `bfs.cc` | BFS | Direction-optimizing breadth-first search |
| `bc.cc` | Betweenness Centrality | Brandes algorithm for BC approximation |
| `cc.cc` | Connected Components | Afforest algorithm for component detection |
| `sssp.cc` | SSSP | Delta-stepping single-source shortest paths |
| `tc.cc` | Triangle Counting | Ordered intersection-based counting |

## Using as Perceptron Features

The feature vector output can be used with GraphBrew's adaptive reordering perceptron:

```cpp
// Feature vector format: [L1_hit, L1_miss, L2_hit, L2_miss, L3_hit, L3_miss, overall_hit, overall_miss]
std::vector<double> features = cache.getFeatureVector();
```

These cache features can help the perceptron predict which reordering algorithm will perform best based on the graph's memory access patterns.

## Architecture

```
bench/
├── include/
│   └── cache_sim/
│       ├── cache_sim.h     # Cache simulator implementation
│       └── graph_sim.h     # Graph wrappers for cache tracking
├── src_sim/
│   ├── pr.cc              # PageRank with cache simulation
│   ├── bfs.cc             # BFS with cache simulation
│   ├── bc.cc              # BC with cache simulation
│   ├── cc.cc              # CC with cache simulation
│   ├── sssp.cc            # SSSP with cache simulation
│   └── tc.cc              # TC with cache simulation
└── bin_sim/               # Compiled simulation binaries
```

## Adding New Algorithms

To add cache simulation to a new algorithm:

1. Include the cache headers:
   ```cpp
   #include "cache_sim/cache_sim.h"
   #include "cache_sim/graph_sim.h"
   ```

2. Wrap graph reads/writes with cache tracking:
   ```cpp
   // Before: NodeID neighbor = g.out_neigh(u, i);
   // After:
   cache.readArray(g.out_neigh_ptr(u), i, sizeof(NodeID), "neighbors");
   NodeID neighbor = g.out_neigh(u, i);
   
   // For property arrays:
   cache.readArray(scores.data(), u, sizeof(ScoreT), "scores");
   ScoreT val = scores[u];
   
   cache.writeArray(scores.data(), u, sizeof(ScoreT), "scores");
   scores[u] = new_val;
   ```

3. Print statistics at the end:
   ```cpp
   cache.printStats();
   cache.exportJSON("stats.json");  // Optional
   ```

## Limitations

- **Performance Overhead**: Cache simulation adds significant overhead (~10-100x slower)
- **Thread Simulation**: Multi-threaded accesses are serialized for accurate tracking
- **Simplified Model**: Does not model cache coherence protocols, prefetching, or speculative access
- **Memory Only**: Does not track instruction cache, TLB, or other CPU structures
