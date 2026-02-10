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
├── include/cache_sim/
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
  "total_accesses": 123456789,
  "memory_accesses": 493827,
  "L1": {
    "size_bytes": 32768,
    "ways": 8,
    "sets": 64,
    "line_size": 64,
    "policy": "LRU",
    "hits": 98765432,
    "misses": 24691357,
    "hit_rate": 0.800000,
    "evictions": 24691000,
    "writebacks": 0
  },
  "L2": {
    "size_bytes": 262144,
    "ways": 4,
    "sets": 1024,
    "line_size": 64,
    "policy": "LRU",
    "hits": 19753086,
    "misses": 4938271,
    "hit_rate": 0.800000,
    "evictions": 4938000,
    "writebacks": 0
  },
  "L3": {
    "size_bytes": 8388608,
    "ways": 16,
    "sets": 8192,
    "line_size": 64,
    "policy": "LRU",
    "hits": 4444444,
    "misses": 493827,
    "hit_rate": 0.900000,
    "evictions": 493000,
    "writebacks": 0
  }
}
```

## Use Cases

### 1. Comparing Reordering Algorithms

Compare cache efficiency across different reorderings:

```bash
# Test ORIGINAL ordering
./bench/bin_sim/pr -f graph.mtx -o 0 -n 1

# Test RabbitOrder (uses csr variant by default)
./bench/bin_sim/pr -f graph.mtx -o 8 -n 1

# Test RabbitOrder with explicit boost variant
./bench/bin_sim/pr -f graph.mtx -o 8:boost -n 1

# Test GraphBrewOrder
./bench/bin_sim/pr -f graph.mtx -o 12 -n 1
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
python3 scripts/graphbrew_experiment.py --full --size small

# Or just the cache phase
python3 scripts/graphbrew_experiment.py --phase cache
```

Cache features are stored in `results/cache_*.json` and integrated into perceptron weights:

```json
{
  "GraphBrewOrder": {
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

# Skip expensive simulations (BC, SSSP) on large graphs
python3 scripts/graphbrew_experiment.py --phase cache --skip-expensive

# Full pipeline includes cache simulation automatically
python3 scripts/graphbrew_experiment.py --full --size medium
```

Results are saved to `results/cache_*.json`:

```json
{
  "graph": "email-Enron",
  "algorithm_id": 12,
  "algorithm_name": "GraphBrewOrder",
  "benchmark": "pr",
  "l1_hit_rate": 85.2,
  "l2_hit_rate": 92.1,
  "l3_hit_rate": 98.5,
  "l1_misses": 0,
  "l2_misses": 0,
  "l3_misses": 0,
  "success": true,
  "error": ""
}
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

All six benchmarks (`pr`, `bfs`, `bc`, `cc`, `sssp`, `tc`) have instrumented versions in `bench/src_sim/`. Key insights:
- **PR**: Pull-based has predictable outgoing edge access
- **BFS**: Direction-optimizing switch point affects cache behavior
- **BC**: Forward/backward passes have different access patterns
- **TC**: Sorted adjacency lists improve intersection cache efficiency

## Limitations

1. **Performance**: ~10-100x slower than uninstrumented code
2. **Threading**: Accesses are serialized for accurate counting
3. **Simplifications**: No cache coherence, prefetching, or speculation modeling
4. **Scope**: Data cache only (no instruction cache or TLB)

## Perceptron Integration

Cache features integrate with the perceptron-based algorithm selector. See [[Perceptron-Weights]] for the full feature vector.

**Cache-specific features:** `cache_l1_impact`, `cache_l2_impact`, `cache_l3_impact`, `cache_dram_penalty` — automatically collected during `--phase cache` and integrated into weight files.

```bash
# Run cache simulation as part of full pipeline
python3 scripts/graphbrew_experiment.py --full --size small

# Or just the cache phase
python3 scripts/graphbrew_experiment.py --phase cache
```

C++ access: `cache_sim::GlobalCache().getFeatures()` or `CACHE_FEATURES()` macro.

## Related Pages

- [[Perceptron-Weights]] - Using cache features for algorithm selection
- [[Correlation-Analysis]] - Correlating cache stats with performance
- [[Benchmark-Suite]] - Running performance experiments

---

[← Back to Home](Home)
