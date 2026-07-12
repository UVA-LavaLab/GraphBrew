# DROPLET vs ECG_PFX — algorithm comparison

**Date:** 2026-06-03
**Sources:**
- DROPLET paper: Basak et al., "Analysis and Optimization of the Memory Hierarchy for Graph Processing Workloads", HPCA 2019.
- DROPLET impl (Sniper): `bench/include/sniper_sim/overlays/common/core/memory_subsystem/parametric_dram_directory_msi/droplet_prefetcher.{h,cc}` (314 LOC)
- DROPLET impl (gem5): `bench/include/gem5_sim/overlays/mem/cache/prefetch/droplet.{hh,cc}`
- ECG_PFX impl (cache_sim): `bench/include/cache_sim/graph_cache_context.h` (encoding) + `bench/src_sim/{pr,bfs,sssp}.cc` (runtime emission)
- cache_sim DROPLET port: `ECG_PREFETCH_MODE=3` sequential lookahead in `bench/src_sim/{pr,bfs,sssp}.cc`

## DROPLET (Basak HPCA'19)

**Key insight:** Graph workloads have two distinct data types:
1. **Edge lists** (CSR adjacency): short reuse, streaming, fits in L1/L2
2. **Property data** (vertex values like scores[v], parent[v], dist[v]): long reuse, irregular, thrashes LLC

**Architecture:** Two decoupled prefetch engines:

```
Edge-list engine          →  Property engine
─────────────────────         ─────────────────────
1. Watches memory             1. Triggered by neighbor IDs
   accesses                      from prefetched edge lines
2. Detects strides            2. Issues property prefetch
   in edge stream                for those vertex IDs
3. Predicts next K edges      3. Decouples from core's
                                 dependency chain
                                 (avoids pointer-chasing stall)
```

**Per-access flow** (from `droplet_prefetcher.cc::getNextAddress`):
1. Check if current address is in edge region → if no, return (no prefetch)
2. Compute line address of current edge access
3. Update stride detector (4-entry stride table, confidence-based)
4. If stride confidence ≥ 2, predict K future edge lines via stride
5. For each predicted edge line:
   - Look up destination vertex IDs (from loaded edge data array)
   - Compute property address per vertex: `property_base + vertex × elem_size`
   - Issue prefetch for property line if not in recent dedup window
6. Also issue indirect property prefetches for **current** edge line's neighbors

**Reported headline:** 1.37× average speedup, 15-45% LLC miss reduction, peak 1.76× on BFS.

**Defaults (per artifact):**
- `prefetch_degree = 1` (one edge-stream line per trigger)
- `indirect_degree = 16` (one 64B cache line of 4B vertex IDs)
- `stride_table_size = 64`

## ECG_PFX (this work)

**Key insight:** Graph access patterns are exposed at compile/encoding time, not just at runtime. The ECG fat-ID can carry a hint per vertex: "the next-touched vertex's property has high reuse — prefetch it."

**Architecture:** Single mask-driven engine with two delivery paths:
1. **Encoded path** (preprocess time): For each vertex `v`, the ECG mask
   stores the ID of `v`'s best-rerefrence neighbor (lowest POPT rank or
   highest degree, configurable). Runtime decodes the PFX bits and
   issues a prefetch on demand.
2. **Runtime lookahead path** (sg_kernel-style): Kernel iterates its
   per-vertex neighbor list and emits `SNIPER_ECG_PFX_TARGET(future_vertex_id)`
   for the K-step lookahead, smart-selected by POPT rank or degree.

**Per-access flow** (from `bench/src_sim/pr.cc` lookahead path):
1. Kernel iterating in-neighbors of vertex u, currently at neighbor v
2. Look at next K in-neighbors (lookahead window)
3. Pick **best target** among those K:
   - mode=1 (degree): highest out-degree → most-popular-future-vertex
   - mode=2 (popt):   lowest POPT rerefence rank → most-future-reused
4. Issue **one** prefetch for `property[best_target]`
5. Continue iteration

**ECG_CONTAINER_BITS = 64** required so PFX gets enough bits to encode
direct vertex IDs for multi-million-vertex graphs (per sprint 6c
finding).

## Key algorithmic differences

| Dimension | DROPLET | ECG_PFX |
|---|---|---|
| Trigger | Stride-detected edge access | Per-iteration kernel hook |
| State | 4-entry stride table + dedup set | None (or POPT rank lookup) |
| Target selection | Sweep next K edges sequentially | Pick best-1 of next K by POPT/degree |
| Prefetches per trigger | indirect_degree (default 16) | 1 |
| Knowledge source | Runtime stride confidence | Compile-time POPT/degree |
| Hardware cost | Stride table + dedup set | Fat-ID PFX bits (per-vertex) |
| Software cost | None (transparent) | Hint emission per iteration |

## cache_sim port equivalence

The cache_sim DROPLET we use for apples-to-apples comparison
(`ECG_PREFETCH_MODE=3`) collapses DROPLET into "sweep next K
in-neighbors sequentially without target selection". This:
- **Same target locations** as ECG_PFX (next K in-neighbors)
- **Same lookahead window** as ECG_PFX (K=8)
- **No smart selection** (sequential, mimicking DROPLET's stride
  prediction issuing K=16 indirect prefetches per trigger)

The original DROPLET's stride-detection state machine is implicit in
the cache_sim version: cache_sim sees the kernel's actual in-neighbor
iteration sequence, so we don't need to detect the stride — we
already have the ground-truth access stream. This is a **best-case
DROPLET** (no stride mis-prediction overhead).

## Measured efficiency (cache_sim, 14 cells)

| Metric | ECG_PFX | DROPLET | Ratio |
|---|---:|---:|---:|
| L3 miss reduction vs LRU | -5.84 pp | -5.84 pp | tied |
| Total requests issued | 145,084,233 | 475,145,125 | **3.27× fewer for ECG_PFX** |
| Total cache fills | 75,642,417 | 211,297,216 | **2.79× fewer for ECG_PFX** |
| Total useful prefetches | 75,634,816 | 211,157,873 | (proportional to fills) |
| Useful_rate | 99.99% | 99.93% | tied |
| **Requests per useful** | **1.918** | **2.250** | **ECG_PFX 15% more efficient** |

## What this means for the paper

ECG_PFX achieves the same cache-miss reduction as DROPLET with
**~1/3 the prefetch bandwidth** consumed. The mechanism is fundamentally
different but the cache-hit benefit is equivalent. Per-cell efficiency
ratio (ECG_PFX req/useful ÷ DROPLET req/useful) ranges from 0.69 to
0.92 — ECG_PFX consistently issues 7-31% fewer prefetch requests per
useful hit.

The bandwidth advantage matters because:
1. **DRAM bandwidth contention** with demand reads
2. **Cache pollution** from over-prefetching (more candidates to evict)
3. **Power consumption** scales with memory traffic

Per-cell ratio table (`ECG_PFX req/useful` ÷ `DROPLET req/useful`,
lower = ECG_PFX more efficient):

| cell | ratio |
|---|---:|
| web-Google/pr | 0.693 |
| soc-pokec/sssp | 0.754 |
| soc-pokec/pr | 0.763 |
| web-Google/bfs | 0.739 |
| cit-Patents/bfs | 0.799 |
| cit-Patents/sssp | 0.819 |
| cit-Patents/pr | 0.868 |
| soc-LiveJournal1/sssp | 0.887 |
| soc-LiveJournal1/bfs | 0.904 |
| soc-LiveJournal1/pr | 0.918 |

Best efficiency win: **web-Google/pr at ratio 0.693 — ECG_PFX issues 30.7% fewer requests per useful hit than DROPLET**.

## Honest caveats

1. Our cache_sim DROPLET is a **best-case** implementation — the real
   DROPLET has stride mis-prediction overhead we don't model. Real-hardware
   DROPLET would issue MORE requests than our model shows.
2. Our cache_sim ECG_PFX gets ground-truth lookahead from the kernel
   (no encoding compression loss). The actual ECG fat-ID encoding has
   PFX bits which can lose precision on graphs > 2^33 vertices
   (`prefetch_direct=false` falls back to hot-table of top-K).
3. Both prefetchers were measured on the same kernel under the same
   eviction policy (ECG_DBG_ONLY) — the comparison is apples-to-apples
   for **algorithm efficiency**, not full-system speedup (which would
   require gem5/Sniper timing).
