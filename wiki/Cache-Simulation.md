# Cache Simulation

The Cache Simulation framework in GraphBrew provides detailed cache performance analysis for graph algorithms. This allows you to understand memory access patterns and cache efficiency at L1, L2, and L3 levels.

## Overview

Graph algorithms are memory-intensive and cache performance significantly impacts execution time. The cache simulation framework instruments graph algorithms to track every memory access, simulating how data flows through a three-level cache hierarchy.

**Key Features:**
- Multi-level cache hierarchy (L1/L2/L3)
- Configurable cache parameters (size, associativity, line size)
- Nine eviction policies (LRU, FIFO, RANDOM, LFU, PLRU, SRRIP, GRASP, P-OPT, ECG)
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
│   ├── cache_sim.h              # Cache simulator core (9 policies, 5 variants)
│   ├── graph_cache_context.h    # Unified graph metadata (GRASP/P-OPT/ECG context)
│   └── graph_sim.h              # Macros: SIM_CACHE_READ, SIM_CACHE_READ_MASKED, SIM_CACHE_READ_EDGE
├── include/graphbrew/partition/cagra/
│   └── popt.h                   # P-OPT rereference matrix builder (makeOffsetMatrix)
├── src_sim/                     # Instrumented algorithms (with P-OPT + CSR tracking)
│   ├── pr.cc                    # PageRank (Gauss-Seidel, CSR edge tracking)
│   ├── pr_spmv.cc               # PageRank SpMV (Jacobi, CSR edge tracking)
│   ├── bfs.cc                   # Breadth-First Search (CSR edge tracking)
│   ├── bc.cc                    # Betweenness Centrality
│   ├── cc.cc                    # Connected Components (Afforest)
│   ├── cc_sv.cc                 # Connected Components (Shiloach-Vishkin)
│   ├── sssp.cc                  # Single-Source Shortest Paths
│   └── tc.cc                    # Triangle Counting
└── bin_sim/                     # Compiled binaries
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
| `ECG_MODE` | DBG_PRIMARY | ECG eviction mode: DBG_PRIMARY, POPT_PRIMARY, DBG_ONLY, ECG_EMBEDDED |
| `ECG_MASK_WIDTH` | 8 | ECG mask bits per edge (2,4,8,16,32) |
| `ECG_NUM_BUCKETS` | 11 | Number of degree buckets (2-16) |
| `ECG_RRPV_BITS` | 3 | RRPV bit width (max RRPV = 2^bits - 1) |
| `ECG_PER_VERTEX` | 0 | Per-vertex masks (1) vs per-edge (0) |
| `ECG_DEGREE_MODE` | OUT | Degree mode: OUT, IN, BOTH |

### Default Cache Configuration

The default configuration models a typical modern CPU:

| Level | Size | Associativity | Lines | Sets |
|-------|------|---------------|-------|------|
| L1 | 32 KB | 8-way | 512 | 64 |
| L2 | 256 KB | 4-way | 4096 | 1024 |

> **Note:** The `FastCacheHierarchy` (per-thread simulation) uses 8-way L2 associativity by default, reflecting modern per-core L2 caches.
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
| **GRASP** | Graph-aware 3-tier RRIP: HOT=1/MODERATE=6/COLD=7 (Faldu et al., HPCA'20) | Power-law graphs with DBG reordering |
| **P-OPT** | Graph-transpose Belady 3-phase: non-property first → oracle distance → RRIP tiebreak (Balaji et al., HPCA'21) | Best miss reduction; requires rereference matrix |
| **ECG** | Layered: GRASP insertion + P-OPT/DBG eviction tiebreak, 4 modes (Mughrabi et al., GrAPL) | Combines structural + oracle; zero-overhead embedded mode |

#### ECG Modes

ECG uses a 3-level layered eviction strategy. All modes start with SRRIP aging (Level 1), then apply mode-dependent tiebreakers:

| Mode | Level 2 | Level 3 | Env Variable |
|------|---------|---------|------|
| **DBG_PRIMARY** (default) | DBG tier (coldest vertex) | Dynamic P-OPT `findNextRef()` | `ECG_MODE=DBG_PRIMARY` |
| **POPT_PRIMARY** | P-OPT 3-phase algorithm (exact match) | DBG tier among P-OPT ties | `ECG_MODE=POPT_PRIMARY` |
| **DBG_ONLY** | DBG tier only | None (fast path) | `ECG_MODE=DBG_ONLY` |
| **ECG_EMBEDDED** | Stored P-OPT hint (from mask, zero LLC overhead) | DBG tier | `ECG_MODE=ECG_EMBEDDED` |

**Insertion RRPV by mode:**
- **DBG_ONLY / DBG_PRIMARY / ECG_EMBEDDED**: GRASP-faithful 3-tier — HOT=1 (P_RRIP), MODERATE=6 (I_RRIP), COLD=7 (M_RRIP)
- **POPT_PRIMARY**: Uniform RRPV=6 for all lines (matching pure P-OPT)

**Non-property data handling:**
- **POPT_PRIMARY**: Phase 0 evicts non-property data (CSR edges, offsets) before property data
- **DBG modes**: Non-property data gets RRPV=7 (cold) at insert, naturally ages out

**Hit promotion:**
- **DBG modes**: Hot region → RRPV=0 (aggressive), others → decrement by 1
- **POPT_PRIMARY**: Universal RRPV=0 (same as SRRIP/P-OPT)

**P-OPT rereference matrix:**
All sim kernels build the P-OPT rereference matrix via `makeOffsetMatrix()` when `CACHE_POLICY=POPT` or `CACHE_POLICY=ECG`. The matrix is a 256-epoch × num_cache_lines compressed oracle.

**CSR edge tracking:**
The PR, BFS, and PR_SPMV kernels track CSR edge list reads via `SIM_CACHE_READ_EDGE()`, providing realistic cache pressure from structure data alongside property data.

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
     L1     ...              ...          ...       XX.XX%      XX.XX%
     L2     ...              ...          ...       XX.XX%      XX.XX%
     L3     ...              ...          ...       XX.XX%      XX.XX%
--------------------------------------------------------------------------------

Overall Statistics:
  Total Memory Accesses: ...
  Total Cache Hits:      ... (XX.XX% of memory)
  Memory Accesses (L3 Misses): ... (XX.XX%)
================================================================================
```

### JSON Export

```bash
CACHE_OUTPUT_JSON=results.json ./bench/bin_sim/pr -g 15 -n 1
```

Output format:
```json
{
  "total_accesses": ...,
  "memory_accesses": ...,
  "L1": {
    "size_bytes": 32768,
    "ways": 8,
    "sets": 64,
    "line_size": 64,
    "policy": "LRU",
    "hits": ...,
    "misses": ...,
    "hit_rate": ...,
    "evictions": ...,
    "writebacks": 0
  },
  "L2": {
    "size_bytes": 262144,
    "ways": 4,
    "sets": 1024,
    "line_size": 64,
    "policy": "LRU",
    "hits": ...,
    "misses": ...,
    "hit_rate": ...,
    "evictions": ...,
    "writebacks": 0
  },
  "L3": {
    "size_bytes": 8388608,
    "ways": 16,
    "sets": 8192,
    "line_size": 64,
    "policy": "LRU",
    "hits": ...,
    "misses": ...,
    "hit_rate": ...,
    "evictions": ...,
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
  "ALGORITHM_NAME": {
    "bias": ...,
    "w_modularity": ...,
    "cache_l1_impact": ...,
    "cache_l2_impact": ...,
    "cache_l3_impact": ...,
    "cache_dram_penalty": ...
  }
}
```

Run `--phase cache` on your graphs to generate actual cache impact values. See [[Perceptron-Weights]] for the full weight schema.

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

All eight benchmarks (`pr`, `pr_spmv`, `bfs`, `cc`, `cc_sv`, `sssp`, `bc`, `tc`) have instrumented versions in `bench/src_sim/` (the two SpMV/SV variants share the base simulation logic). Key insights:
- **PR**: Pull-based has predictable outgoing edge access
- **BFS**: Direction-optimizing switch point affects cache behavior
- **BC**: Forward/backward passes have different access patterns
- **TC**: Sorted adjacency lists improve intersection cache efficiency

## Limitations

1. **Performance**: ~10-100x slower than uninstrumented code (CSR edge tracking doubles access count)
2. **Threading**: Use `OMP_NUM_THREADS=1` for accurate single-threaded results. Per-thread AccessHints are thread-safe, but cache data structures are shared and not mutex-protected — multi-threaded runs may produce data races in cache sets
3. **Simplifications**: No cache coherence, hardware prefetching, or speculation modeling
4. **Scope**: Data cache only (no instruction cache or TLB)
5. **CSR tracking**: PR, BFS, PR_SPMV track edge list reads; other kernels (BC, CC, SSSP) track property arrays only

## Validation Results

The cache policies have been validated against the original reference implementations:

**GRASP** (Faldu et al., HPCA'20): Faithful 3-tier implementation matching the [original repo](https://github.com/faldupriyank/grasp). GRASP/LRU ratio = 0.65 (paper: 0.64) on web-Google PR at 1MB LLC.

**P-OPT** (Balaji et al., HPCA'21): Algorithm 2 three-phase eviction matching the [original repo](https://github.com/CMUAbstract/POPT-CacheSim-HPCA21). Achieves 40% miss reduction vs LRU.

**ECG mode equivalences** (validated on web-Google PR, 1MB LLC, OMP=1):
- ECG(DBG_ONLY) ≈ GRASP: 0.1-1.5% relative difference
- ECG(POPT_PRIMARY) ≈ P-OPT: 0.1% relative difference
- P-OPT ≤ ECG(DBG_PRIMARY) ≤ GRASP: hierarchy holds

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
