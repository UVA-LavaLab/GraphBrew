# DROPLET — Data-awaRe decOuPLed prEfeTcher for Graphs

## Citation

```bibtex
@inproceedings{basak_analysis_2019,
  title     = {Analysis and Optimization of the Memory Hierarchy for Graph Processing Workloads},
  author    = {Basak, Abanti and Li, Shuangchen and Hu, Xing and Oh, Sang Min and Xie, Xinfeng and Zhao, Li and Jiang, Xiaowei and Xie, Yuan},
  booktitle = {2019 IEEE International Symposium on High-Performance Computer Architecture (HPCA)},
  pages     = {373--386},
  month     = feb,
  year      = {2019},
  doi       = {10.1109/HPCA.2019.00051},
  note      = {ISSN: 2378-203X}
}
```

## Author & Repository

- **First Author**: Abanti Basak ([@abasak24](https://github.com/abasak24)), UC Santa Barbara, Scalable Energy-Efficient Architecture Lab (SEAL)
- **Advisors**: Prof. Yuan Xie, Prof. Yufei Ding
- **Paper PDF**: [seal.ece.ucsb.edu](https://seal.ece.ucsb.edu/sites/default/files/publications/hpca-2019-abanti.pdf)
- **Slides**: [abasak24.github.io/slides/hpca2019_droplet.pdf](https://abasak24.github.io/slides/hpca2019_droplet.pdf)
- **Source code**: Modified Sniper-6.1 simulator (original repo was public, now deleted from GitHub). A local copy of the DROPLET source is preserved at `d:/github_repos/DROPLET-master/`
- **Simulator**: Sniper-6.1 multi-core architecture simulator with modifications to `common/` and `config/` directories
- **Related repo**: [abasak24/SAGA-Bench](https://github.com/abasak24/SAGA-Bench) — streaming graph analytics benchmarking (ISPASS 2020)
- **Cited by**: 119+ (as of 2026)

## Source Code Structure (from preserved copy)

```
DROPLET-master/
├── common/core/memory_subsystem/
│   ├── cache/
│   │   ├── cache_set_srrip.cc/.h     — SRRIP replacement policy (M-bit RRPV)
│   │   ├── cache_set_lru.cc/.h       — LRU baseline
│   │   └── cache_set_plru.cc/.h      — Pseudo-LRU
│   └── parametric_dram_directory_msi/
│       ├── dropletL1.cc/.h            — DROPLET L1 stream prefetcher (structure-aware)
│       ├── address_prefetcher.cc/.h   — Address-based indirect prefetcher (key DROPLET component)
│       ├── graph_stream_prefetcher.cc/.h — Graph-aware stream prefetcher
│       ├── baseline_stream_prefetcher.cc/.h — Baseline stream prefetcher
│       └── cache_cntlr.cc/.h          — Cache controller with data-type routing
├── config/
│   ├── prefetcher.cfg                 — Prefetcher configurations (6 prefetcher types)
│   ├── gainestown.cfg                 — Default CPU model (Gainestown Xeon)
│   └── rob.cfg                        — ROB configuration
└── example_runme.sh                   — Example: 4-core, Gainestown, SRRIP, ROI-limited
```

### Key Implementation Files

**`dropletL1.cc/.h`** — DROPLET's L1 stream prefetcher:
- Implements stream detection with N streams (default 64), per-core
- 3-step stream confirmation before activating prefetch
- Monitor region tracking: `[startAddress, endAddress]` per stream
- Prefetch degree and distance are configurable (default degree=1, distance=16)
- Page-boundary-aware: stops prefetching at page boundaries

**`address_prefetcher.cc/.h`** — The core DROPLET indirect prefetcher:
- Reads graph structure address ranges from `StrucAddress.txt` (software communication)
- Classifies memory accesses: `isStructureCacheline()` → distinguishes edge list from other data
- **Indirect prefetch chain**: When edge list cache line arrives, extracts neighbor IDs (4B each), computes offset array addresses, then prefetches corresponding structure cache lines
- Handles directed and undirected graphs (separate in/out offset/neighbor arrays)
- Prefetch distance = 4 entries ahead within each cache line (16 IDs per 64B line)
- Caps structure prefetches at 4 cache lines per neighbor to limit bandwidth

**`cache_set_srrip.cc/.h`** — SRRIP cache replacement:
- Configurable M-bit RRPV (from config: `srrip/bits`)
- Insert at RRPV = `m_rrip_max - 1` (long re-reference)
- Hit → RRPV = 0
- Eviction: scan for RRPV = max, age all if not found
- Includes QBS (Query-Based Selection) for inclusive hierarchies

## Why This Paper Matters for ECG

DROPLET is a **key baseline** for graph prefetching and cache optimization. It provides:

1. **In-depth memory hierarchy characterization**: The paper's data-type-aware analysis of graph workloads is the foundation for understanding *why* graph algorithms have poor cache behavior — load-load dependency chains between different data types (vertex properties, edge lists, auxiliary arrays) form the primary bottleneck
2. **Data-type-aware prefetching**: DROPLET distinguishes between different memory regions (property data, edge lists, auxiliary buffers) — ECG adopts this same multi-region awareness in its MASK encoding
3. **L2 cache insight**: The paper shows that private L2 caches have *negligible* contribution to graph performance, while the shared L3 shows high sensitivity — this justifies ECG's focus on LLC-level optimization
4. **Decoupled prefetching**: DROPLET separates prefetching into independent engines per data type — ECG's prefetch hint bits in the fat-ID serve a similar purpose (hinting which vertex to prefetch) but with zero hardware overhead

If our ECG cache policy claims to beat or match DROPLET's prefetching benefits via software-level hints embedded in vertex IDs, we need to demonstrate this against DROPLET as a baseline.

## Key Contributions

### Contribution 1: Data-Type-Aware Memory Characterization

The paper performs the first systematic characterization of memory hierarchy behavior for graph workloads, distinguishing between application data types:

- **Property data (vertex arrays)**: Irregular access via neighbor IDs — poor spatial locality, high miss rate
- **Edge list (CSR indices)**: Semi-sequential streaming — good spatial locality within a vertex's neighbor list
- **Auxiliary data (distances, PageRank values)**: Working data — access pattern depends on algorithm
- **Index arrays (offsets)**: Sequential access with good locality

**Key finding**: Load-load dependency chains between data types form the primary bottleneck for memory-level parallelism (MLP). Specifically:
```
Edge list load → produces neighbor_id → Property data load (dependent)
```
This dependency chain limits out-of-order core's ability to issue multiple outstanding memory requests. The effective MLP is only 2-4x for graph workloads vs 10-20x for streaming workloads.

### Contribution 2: Heterogeneous Reuse Distance Analysis

Different graph data types exhibit very different reuse distances:

| Data Type | Reuse Distance | Cache Behavior |
|-----------|---------------|----------------|
| Property data (vertex) | Long, irregular | Thrashes L2, contributes to L3 misses |
| Edge list (CSR) | Short, streaming | Fits in L1/L2, low miss rate |
| Index arrays | Very short | Fits in L1 |
| Auxiliary data | Algorithm-dependent | Varies by workload |

**Key insight**: The private L2 cache has **negligible** performance contribution for graph workloads — 95%+ of L2 capacity is wasted on property data that will be evicted before reuse. In contrast, the shared L3 cache shows **high performance sensitivity** because it aggregates reuse from multiple cores.

### Contribution 3: DROPLET Prefetcher

DROPLET = **D**ata-awa**R**e dec**O**u**PL**ed pr**E**fe**T**cher

A hardware prefetcher designed specifically for graph workloads:

**Architecture:**
```
Traditional prefetcher:  One prefetch engine → all data types
DROPLET:                 Separate prefetch engines per data type:
                           Engine 1: Property data (vertex arrays) — indirect prefetch
                           Engine 2: Edge list (CSR neighbor arrays) — streaming prefetch
                           Engine 3: Auxiliary arrays — stride/streaming prefetch
```

**Key mechanism — Indirect Prefetching for Property Data:**

The critical bottleneck is the load-load dependency chain:
```
1. Load edge_list[offset + i]  → gets neighbor_id
2. Load property[neighbor_id]  → DEPENDENT on step 1 (cache miss)
```

DROPLET breaks this by:
1. **Look-ahead**: Prefetch edge list entries ahead of the current processing point
2. **Indirect resolve**: When edge list data arrives, immediately issue prefetch for the corresponding property data address
3. **Decoupling**: The property prefetch engine operates independently, filling the cache with property data before the core requests it

```
Core processing:       vertex v, neighbor i
Edge prefetch engine:  already fetched edge_list[offset + i+K]  (K entries ahead)
Property prefetch:     already issued property[neighbor_id(i+K)]
Result:                property[neighbor_id] is in cache when core needs it
```

**DROPLET performance** (from the paper):
- **1.37x** average speedup over no prefetching baseline
- **Up to 1.76x** on BFS workloads
- Reduces LLC miss rate by **15-45%** depending on workload and graph
- Most effective on traversal algorithms (BFS, SSSP) where the access frontier is predictable

### Paper's Benchmarks & Evaluation

- **Simulator**: Sniper-6.1 (modified — source preserved locally)
- **Core model**: 4-core OOO Gainestown (Intel Xeon), 128-entry ROB, LLL cutoff=35
- **Cache hierarchy**: L1=32KB/8-way, L2=256KB/8-way (private), L3=8MB/16-way (shared, SRRIP)
- **DRAM**: 30 GB/s per-controller bandwidth
- **Simulation**: ROI-limited, 600M instructions per run (`stop-by-icount:600000000`)
- **Benchmarks**: GAP Benchmark Suite — PageRank (PR), BFS, Connected Components (CC), SSSP
- **Graphs**: soc-LiveJournal1, soc-Pokec, web-Google, USA-Road, cit-Patents (Table III in paper)
- **Prefetcher configs** (from `config/prefetcher.cfg`):
  - `baseline_stream`: 64 streams, degree=1, distance=16, page-boundary-aware
  - `graph_stream`: 64 streams, graph-aware stream detection
  - `address_prefetch`: Indirect prefetch via `StrucAddress.txt` communication
  - `dropletL1`: Decoupled L1 prefetcher combining stream + address-indirect
- **Run command**: `run-sniper -n 4 -c gainestown -c rob -c prefetcher -g perf_model/core/interval_timer/lll_cutoff=35 -g perf_model/dram/per_controller_bandwidth=30 -s stop-by-icount:600000000 --roi -- <benchmark> -f <graph.sg> -n 1`

## Relationship to ECG

| Aspect | DROPLET | ECG |
|--------|---------|-----|
| **Approach** | Hardware prefetcher | Cache replacement + embedded metadata |
| **Hardware cost** | Dedicated prefetch engines (area) | 6 tag bits/line (~96KB SRAM) |
| **Prefetch mechanism** | Look-ahead + indirect resolve | Fat-ID prefetch hint bits |
| **Cache policy** | Standard (LRU/SRRIP) | 3-level tiebreaking (DBG+P-OPT) |
| **Data-type awareness** | Per-engine specialization | Multi-region classification in MASK |
| **Reorder dependency** | None | DBG or community reordering |
| **Complementary?** | **Yes** — DROPLET prefetches, ECG manages what stays in cache |

**Key insight**: DROPLET and ECG are **complementary**, not competing. DROPLET brings data into the cache faster (prefetch), while ECG decides what to keep and what to evict (replacement). An ideal system would use both:
- DROPLET-style prefetching to fill cache with property data ahead of time
- ECG's degree-aware RRPV + embedded oracle to manage eviction

ECG's prefetch hint bits in the fat-ID are a **software-level approximation** of DROPLET's hardware indirect prefetching — they encode which "hot" neighbor to prefetch next, achieving a subset of DROPLET's benefit at zero hardware cost.

## Related Work from Same Group

- **SAGA-Bench** (Basak et al., ISPASS 2020): Software and hardware characterization of *streaming* graph analytics workloads. [GitHub](https://github.com/abasak24/SAGA-Bench)
- **GraphPIM** (Nai et al., HPCA 2017): Processing-in-memory for graph workloads
- **HyGCN** (Yan et al., MICRO 2019): Alleviating irregularity in graph analytics acceleration — hardware/software co-design

## GraphBrew Integration

- **BibTeX key**: `basak_analysis_2019` (already in ECG `references.bib`)
- **Already cited** in ECG draft introduction: `\cite{basak_analysis_2019}` 
- **Relevance to experiments**: 
  - Section D4 (Prefetch Integration) should compare ECG's prefetch hints against DROPLET's reported numbers
  - Section C2 (Capacity Savings) can reference DROPLET's finding that L2 has negligible contribution — justifying ECG's focus on LLC
  - DROPLET's data-type-aware characterization supports ECG's multi-region MASK encoding design
