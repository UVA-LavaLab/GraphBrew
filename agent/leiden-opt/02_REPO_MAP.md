# Repo Map — Leiden Optimization

Key files and call chains relevant to Leiden variant development.

---

## Core Implementation Files

### Leiden Algorithm (Community Detection)

| File | Lines | Purpose |
|------|:---:|---------|
| `bench/include/graphbrew/reorder/reorder_leiden.h` | 7728 | GVE-Leiden: local-moving, refinement, aggregation, dendrogram |
| `bench/include/external/leiden/leiden.hxx` | 1589 | Reference leiden.hxx implementation (Traag et al.) |

**Key functions in `reorder_leiden.h`:**
- `leidenCSRLocalMoving()` (~L355) — Phase 1: greedily move vertices to best community
- `leidenCSRRefinement()` (~L469) — Phase 2: Leiden refinement (singleton constraint)
- `leidenCSRRefinementFlat()` (~L549) — Flat-array optimized refinement
- `gveLeidenCSR()` (~L806) — Full pipeline: local-moving + refinement + aggregate
- `gveLeidenCSROpt()` (~L923) — Cache-optimized variant with prefetching
- `GVELeidenRefinementPhase()` (~L1018) — Exact-match to paper Algorithm 3
- `GVELeidenRefinementPhaseFlatHash()` (~L1184) — Hash-based flat refinement

### VIBE Framework (Modular Variant System)

| File | Lines | Purpose |
|------|:---:|---------|
| `bench/include/graphbrew/reorder/reorder_vibe.h` | 7055 | VIBE: all ordering strategies, aggregation, community detection |

**Key types in `reorder_vibe.h`:**
- `VibeAlgorithm` enum (~L231): `LEIDEN` or `RABBIT_ORDER`
- `AggregationStrategy` enum (~L237): `LEIDEN_CSR`, `RABBIT_LAZY`, `HYBRID`
- `OrderingStrategy` enum (~L243): 13 strategies (BFS, DFS, DBG, CORDER, HRAB, TQR, etc.)
- `CommunityMode` enum (~L260): `FULL_LEIDEN`, `FAST_LP`, `HYBRID`
- `VibeConfig` struct (~L273): All tunable parameters
- `generateVibeMapping()` (~L6448): Main entry point

**Key parameters in `VibeConfig` (~L273):**
- `resolution` — modularity resolution [0.1–2.0]
- `tolerance` — convergence threshold
- `maxIterations` — per-pass iteration limit
- `maxPasses` — aggregation pass limit
- `useRefinement` — enable/disable Leiden refinement step
- `useGorderIntra` — Gorder-greedy within communities (vibe:hrab:gordi)
- `gorderWindow` — sliding window size for Gorder intra-community
- `useHubExtraction` — extract high-degree hubs before ordering
- `useDynamicResolution` — per-pass resolution adjustment

### GraphBrew Order (Cluster + Per-Community Reorder)

| File | Lines | Purpose |
|------|:---:|---------|
| `bench/include/graphbrew/reorder/reorder_graphbrew.h` | 928 | GraphBrewOrder (algo 12): cluster variant selection + per-community reorder |

**Key types:**
- `GraphBrewCluster` enum (~L203): `Leiden`, `GVE`, `GVEOpt`, `Rabbit`, `HubCluster`
- `GraphBrewConfig` struct (~L335): cluster variant, final algo, resolution, levels
- CLI format: `-o 12:cluster_variant:final_algo:resolution:levels`

### RabbitOrder (The Competitor)

| File | Lines | Purpose |
|------|:---:|---------|
| `bench/include/graphbrew/reorder/reorder_rabbit.h` | 1161 | RabbitOrder: hierarchical Louvain + parallel incremental aggregation |

- Variants: `csr` (native, no deps), `boost` (original, needs Boost)
- CLI: `-o 8` or `-o 8:csr` or `-o 8:boost`

---

## Cache Simulation

| File | Lines | Purpose |
|------|:---:|---------|
| `bench/include/cache_sim/cache_sim.h` | 1909 | Full L1/L2/L3 cache hierarchy simulator |
| `bench/include/cache_sim/graph_sim.h` | 77 | Graph-specific simulation wrappers |

**Cache hierarchy defaults:**
- L1: 32KB, 8-way associative, 64B lines
- L2: 256KB, 4-way
- L3: 8MB, 16-way (shared)
- Configurable via environment variables: `CACHE_L1_SIZE`, `CACHE_L3_WAYS`, `CACHE_POLICY`

**Eviction policies:** LRU, FIFO, RANDOM, LFU, PLRU, SRRIP

**Metrics:** per-level hit/miss rates, total access counts, JSON output

---

## Benchmark & Experiment Infrastructure

| File | Lines | Purpose |
|------|:---:|---------|
| `scripts/graphbrew_experiment.py` | — | Main experiment driver (use for ALL evaluation) |
| `scripts/lib/reorder.py` | 1204 | CSR variant registration and reorder dispatch |
| `scripts/lib/benchmark.py` | 728 | Benchmark execution (PR, BFS, CC, SSSP, BC, TC) |
| `scripts/lib/cache.py` | 674 | Cache simulation integration |
| `scripts/lib/build.py` | 394 | Build system integration |
| `scripts/lib/phases.py` | 1094 | Experiment phase orchestration |
| `scripts/lib/results.py` | — | Result collection and reporting |

**Python variant registration** (`scripts/lib/reorder.py`):
All VIBE sub-variants are registered here. When you add a new variant in C++,
you must also register it in this file for the experiment pipeline to find it.

---

## Types & Shared Infrastructure

| File | Lines | Purpose |
|------|:---:|---------|
| `bench/include/graphbrew/reorder/reorder_types.h` | 4681 | Shared types, enums, PerceptronWeights, feature computation |
| `bench/include/graphbrew/reorder/reorder.h` | — | Top-level dispatcher (routes algo ID to implementation) |
| `bench/include/graphbrew/reorder/reorder_basic.h` | — | Basic algos: RANDOM, SORT, HubSort, DBG, etc. |

---

## Call Chain: LeidenCSR Variant Execution

```
graphbrew_experiment.py
  → scripts/lib/reorder.py         (parse variant string, build CLI args)
  → bench/bin/pr -o 17:gveopt2     (binary invocation)
    → reorder.h                    (dispatch to LeidenCSR handler)
      → reorder_leiden.h           (GVE-Leiden pipeline: local-move → refine → aggregate)
      → reorder_vibe.h             (if VIBE variant: parse sub-variant → configure → run)
        → VibeConfig::FromString() (parse "vibe:hrab:gordi" → set all flags)
        → generateVibeMapping()    (community detection → ordering → mapping)
```

## Call Chain: Cache Simulation

```
graphbrew_experiment.py --cache-sim
  → scripts/lib/cache.py           (invoke cache-simulated benchmark binary)
  → bench/src_sim/pr.cc            (PageRank with cache instrumentation)
    → cache_sim.h                  (CacheHierarchy: L1 → L2 → L3 simulation)
    → graph_sim.h                  (GraphCacheSim: access pattern recording)
  → results JSON                   (per-level hit/miss rates)
```
