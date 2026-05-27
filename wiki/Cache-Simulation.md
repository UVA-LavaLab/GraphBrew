# Cache Simulation

Trace-driven cache simulator for graph kernels. Instruments each
memory access in PR / BFS / CC / SSSP / BC / TC and reports L1 / L2 / L3
hit rates under nine eviction policies. Use it to compare reorderings
on a fixed cache geometry without machine noise — the metric is
deterministic per process.

## Quick start

```bash
make all-sim
./bench/bin_sim/pr -f graph.sg -s -n 1 -o 12:hrab

# Custom cache geometry / policy
CACHE_L3_SIZE=1048576 CACHE_POLICY=SRRIP \
    ./bench/bin_sim/pr -f graph.sg -s -n 1 -o 12:hrab
```

The summary block in the output reports `Memory Accesses` and
`Overall Hit Rate` — these are the headline numbers for cache-quality
comparisons.

## Overview

Real wallclock kernel time mixes cache effects with TLB pressure,
branch prediction, scheduler noise, and OS jitter. The simulator
isolates cache behaviour by replaying every load address through a
deterministic three-level cache hierarchy and reports:

- per-level hits / misses / hit-rate
- overall memory accesses
- per-policy eviction stats (RRPV histograms for SRRIP / GRASP / P-OPT / ECG)
- optional JSON export for downstream analysis

Output is deterministic for a given (graph, ordering, policy, cache
size) tuple, so a single run suffices — there is no noise to average out.

## Architecture

```
bench/
├── include/cache_sim/
│   ├── cache_sim.h              # Cache simulator core policies and graph-aware modes
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
| **GRASP** | Graph-aware 3-tier RRIP: HIGH insert=1 / MODERATE=6 / LOW=7, hot hit=0 (Faldu et al., HPCA'20) | Power-law graphs with DBG reordering |
| **P-OPT** | Graph-transpose Belady 3-phase: non-property first -> oracle distance -> RRIP tiebreak (Balaji et al., HPCA'21) | Oracle ceiling; requires rereference matrix |
| **P-OPT charged** | Runner alias `POPT_CHARGED`: same dynamic P-OPT logic, but effective LLC data ways are reduced for rereference matrix columns | Honest P-OPT prior-method baseline |
| **ECG** | Layered: GRASP insertion + P-OPT/DBG eviction tiebreak, multiple modes (Mughrabi et al., GrAPL) | Combines structural + oracle; zero-overhead embedded mode |

#### ECG Modes

ECG uses a 3-level layered eviction strategy. All modes start with SRRIP aging (Level 1), then apply mode-dependent tiebreakers:

| Mode | Level 2 | Level 3 | Env Variable |
|------|---------|---------|------|
| **DBG_PRIMARY** (default) | DBG tier (coldest vertex) | Dynamic P-OPT `findNextRef()` | `ECG_MODE=DBG_PRIMARY` |
| **POPT_PRIMARY** | P-OPT 3-phase algorithm (exact match) | DBG tier among P-OPT ties | `ECG_MODE=POPT_PRIMARY` |
| **POPT_TIE** | SRRIP max-RRPV candidates | Dynamic P-OPT, then DBG tier | `ECG_MODE=POPT_TIE` |
| **DBG_ONLY** | DBG tier only | None (fast path) | `ECG_MODE=DBG_ONLY` |
| **ECG_EMBEDDED** | Stored P-OPT hint (from mask, zero LLC overhead) | DBG tier | `ECG_MODE=ECG_EMBEDDED` |
| **ECG_EPOCH_EMBEDDED** | Current-epoch compact P-OPT hint, cache_sim experimental mode | DBG tier | `ECG_MODE=ECG_EPOCH_EMBEDDED` |
| **ECG_COMBINED** | Combined DBG + stored P-OPT insertion RRPV | None | `ECG_MODE=ECG_COMBINED` |

**Insertion RRPV by mode:**
- **DBG_ONLY / DBG_PRIMARY / ECG_EMBEDDED**: GRASP-faithful 3-tier — HIGH insert=1 (P_RRIP), MODERATE=6 (I_RRIP), LOW=7 (M_RRIP)
- **POPT_PRIMARY**: Uniform RRPV=6 for all lines (matching pure P-OPT)

**Non-property data handling:**
- **POPT_PRIMARY**: Phase 0 evicts non-property data (CSR edges, offsets) before property data
- **DBG modes**: Non-property data gets RRPV=7 (cold) at insert, naturally ages out

**Hit promotion:**
- **DBG modes**: Hot-region hits → RRPV=0 (aggressive), other hits → decrement by 1
- **POPT_PRIMARY**: Universal RRPV=0 (same as SRRIP/P-OPT)

**P-OPT rereference matrix:**
All sim kernels build the P-OPT rereference matrix via `makeOffsetMatrix()` when `CACHE_POLICY=POPT` or `CACHE_POLICY=ECG`. The matrix is a 256-epoch × num_cache_lines compressed oracle.

`POPT_CHARGED` and `ECG:*_CHARGED` are experiment-runner labels, not direct
`CACHE_POLICY` values. Use them through `scripts/experiments/ecg/roi_matrix.py`
or `scripts/experiments/ecg/final_paper_run.py`. Charged rows reserve effective
LLC data ways for current+next rereference matrix columns and report matrix
streaming fields in CSV output. See [[ECG-Final-Runs]].

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

Run `--phase cache` on your graphs to generate actual cache impact values. See [[AdaptiveOrder-ML]] for the full weight schema.

The perceptron uses these weights to factor cache performance into algorithm selection.

### 4. Automated Cache Benchmark Suite

Run cache benchmarks using the unified script:

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

**GRASP** (Faldu et al., HPCA'20): Faithful 3-tier implementation with high-reuse insertion at RRPV=1, high-reuse hit promotion to 0, moderate insertion at 6, low insertion at 7, and plain SRRIP victim selection. This matches the official `faldupriyank/grasp` `P_RRIP=1` / `H_RRIP=0` split.

**P-OPT** (Balaji et al., HPCA'21): Algorithm 2 three-phase eviction using an explicit current-vertex signal. Use `POPT` as the uncharged oracle ceiling and `POPT_CHARGED` as the overhead-aware prior-method baseline.

**Current focused validation:**
- 2026-05-27 source-faithfulness audit: cache_sim, gem5 overlays, and Sniper overlays now use GRASP hot insertion RRPV=1 / hot hit RRPV=0, and P-OPT mixed sets use upstream Phase 1 non-property eviction rather than the earlier far-rereference boost heuristic.
- cache_sim component proof at `results/ecg_experiments/proof_matrix/component_g12_l3_4kb_grasp_rrpv0/proof_matrix.csv` predates the RRPV=1 correction and should be refreshed before use in final claims.
- gem5 GRASP parity: `GRASP` and `ECG_DBG_ONLY` should be refreshed on PR/BFS/SSSP after the RRPV=1 correction; the old RRPV=0 parity rows are no longer final evidence.
- gem5 P-OPT current-vertex validation: PR/SSSP selected rows pass with explicit `GEM5_SET_VERTEX` hints.
- charged P-OPT smoke: `results/ecg_experiments/roi_matrix/popt_charged_gem5_smoke/roi_matrix.csv` shows `POPT_CHARGED` and `ECG_DBG_PRIMARY_CHARGED` run end-to-end with visible reserved-way overhead.

## Perceptron Integration

Cache features integrate with the perceptron-based algorithm selector. See [[AdaptiveOrder-ML]] for the full feature vector.

**Cache-specific features:** `cache_l1_impact`, `cache_l2_impact`, `cache_l3_impact`, `cache_dram_penalty` — automatically collected during `--phase cache` and integrated into weight files.

```bash
# Run cache simulation as part of full pipeline
python3 scripts/graphbrew_experiment.py --full --size small

# Or just the cache phase
python3 scripts/graphbrew_experiment.py --phase cache
```

C++ access: `cache_sim::GlobalCache().getFeatures()` or `CACHE_FEATURES()` macro.

## Related Pages

- [[AdaptiveOrder-ML]] - Using cache features for algorithm selection
- [[Benchmark-Suite]] - Correlating cache stats with performance
- [[Benchmark-Suite]] - Running performance experiments

---

[← Back to Home](Home)
