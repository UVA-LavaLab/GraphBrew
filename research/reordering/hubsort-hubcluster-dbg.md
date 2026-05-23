# HubSort, HubCluster, DBG — Algorithm IDs: 3, 4, 5, 6, 7

## Citations

```bibtex
@inproceedings{dbg-iiswc19,
  title     = {A Closer Look at Lightweight Graph Reordering},
  author    = {Faldu, Priyank and Diamond, Jeff and Grot, Boris},
  booktitle = {International Symposium on Workload Characterization (IISWC)},
  year      = {2019},
  month     = nov
}

@inproceedings{iiswc18,
  title     = {When is Graph Reordering an Optimization? Studying the Effect of Lightweight Graph Reordering Across Applications and Input Graphs},
  author    = {Balaji, Vignesh and Lucia, Brandon},
  booktitle = {International Symposium on Workload Characterization (IISWC)},
  year      = {2018},
  month     = sep
}
```

## Official Repositories

- **DBG**: [faldupriyank/dbg](https://github.com/faldupriyank/dbg) — Apache-2.0
  - Multi-threaded DBG, Hub Clustering, Hub Sorting, Sort, Random (in `ligra/dbg.h`)
  - Tested on x86 with gcc 6.4.0 on Ubuntu 14.04.1
  - Benchmarks: PageRank, PRDelta, BellmanFord, BC, Radii on web-Google + others
- **IISWC'18 Packing Factor**: [CMUAbstract/Graph-Reordering-IISWC18](https://github.com/CMUAbstract/Graph-Reordering-IISWC18) — MIT
  - Hub Sorting, Hub Clustering, Packing Factor computation
  - By Vignesh Balaji ([@bvignesh](https://github.com/bvignesh))
  - References: Zhang et al. "Making Caches Work for Graph Analytics" (COrder)

## Why Faithful Implementation Matters

DBG (Algorithm 5) is **critical infrastructure** for both reordering and caching:
- **GRASP** uses DBG vertex grouping to set RRPV insertion priorities
- **ECG** uses DBG annotations in its tiebreaking hierarchy
- ECG Section A1 validates GRASP's dependence on DBG — if DBG bucket boundaries change, A1 invariants break
- Hub-based algorithms (3-7) are baselines in VLDB experiments

## Algorithm Descriptions

### HubSort (Algo 3)
Sort vertices by descending degree. Hubs get low IDs → packed at array front → stay in cache.
**O(|V| log |V|)**

### HubCluster (Algo 4)
1. Identify hubs (top 1-10% by degree)
2. Assign contiguous IDs to hubs, then to hub neighbors, then remaining vertices.
**O(|V| + |E|)**

### DBG — Degree-Based Grouping (Algo 5)
Groups vertices into **8 degree-based buckets** with thresholds at multiples of the average degree: `[0, avg/2, avg, avg×2, avg×4, avg×8, avg×16, avg×32, ∞)`. Vertices within each bucket get contiguous IDs. Highest-degree bucket first.

Note: The cache simulator (`graph_cache_context.h`) independently uses **11 topology buckets** for RRPV classification — this is a separate system from the 8-bucket reordering.

**O(|V|)** with bucket sort.

### HubSortDBG (Algo 6) / HubClusterDBG (Algo 7)
Compositions: HubSort+DBG or HubCluster+DBG for additive benefits.

## GraphBrew Integration

- **Algorithm IDs**: 3-7
- **C++ Implementation**: `bench/include/graphbrew/reorder/reorder_hub.h`
- **CLI**: `-o 3` through `-o 7`
- **Used by ECG**: DBG (Algo 5) is the graph-structure annotation for GRASP and ECG cache policies
- **Used by GRASP**: DBG groups → RRPV insertion priorities
