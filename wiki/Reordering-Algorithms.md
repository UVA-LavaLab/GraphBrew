# Reordering Algorithms

GraphBrew implements 17 algorithm IDs (`-o 0` through `-o 16`).
Two are baselines (no useful reordering), one is a runtime selector
(AdaptiveOrder, research-only), one loads a precomputed permutation
from disk (MAP). The remaining 13 produce orderings you can benchmark.

## Why reorder

Power-law and clustered graphs have neighbour-access patterns that
miss in cache when vertices are placed randomly. Renumbering vertices
so that frequently co-accessed ones land on nearby cache lines turns
unpredictable misses into hits. Reordering does **not** change the
graph's topology — only the integer labels.

Three orthogonal locality dimensions matter:

| Dimension | Captured by | Example algorithms |
|---|---|---|
| Spatial (community structure) | Leiden, Rabbit Order | `12:leiden`, `12:rabbit`, RABBIT (8) |
| Temporal (degree skew) | hub grouping | HUBCLUSTER (4), DBG (5) |
| Convergence (forward-edge fraction) | edge-direction optimisation | GoGraph (16), chained `12:leiden → 16` |

GraphBrew (`-o 12`) is the framework that composes these dimensions;
the other IDs are individual primitives or baselines.

## Quick reference

| ID | Flag | Algorithm | Complexity | Notes |
|---|---|---|---|---|
| 0 | `-o 0` | ORIGINAL | O(1) | input ordering, baseline |
| 1 | `-o 1` | RANDOM | O(n) | random permutation, worst-case baseline |
| 2 | `-o 2` | SORT | O(n log n) | degree-descending sort |
| 3 | `-o 3` | HUBSORT | O(n log n) | hubs first (full degree sort) |
| 4 | `-o 4` | HUBCLUSTER | O(n) | hubs first, non-hubs keep original order |
| 5 | `-o 5` | DBG | O(n) | logarithmic degree buckets |
| 6 | `-o 6` | HUBSORTDBG | O(n log n) | DBG buckets + degree sort within |
| 7 | `-o 7` | HUBCLUSTERDBG | O(n) | DBG buckets + hub-cluster within |
| 8 | `-o 8` | RABBITORDER | O(n log n + m) | Louvain + dendrogram DFS; variants `csr` (default), `boost` |
| 9 | `-o 9` | GORDER | O(n·w + m) | sliding-window greedy, w=5; high cache quality, slow reorder |
| 10 | `-o 10` | CORDER | O(n log n) | cache-aware bandwidth reduction |
| 11 | `-o 11` | RCM | O(n log n + m) | Reverse Cuthill–McKee; variants `default`, `bnf` |
| 12 | `-o 12` | GraphBrewOrder | O(n log n + m) | composable pipeline — see [GraphBrewOrder](GraphBrewOrder) |
| 13 | `-o 13:<file>` | MAP | O(n) | load permutation from `.lo` / `.so` file |
| 14 | `-o 14` | AdaptiveOrder | varies | ML selector; research-only, see [AdaptiveOrder-ML](AdaptiveOrder-ML) |
| 15 | `-o 15` | LeidenOrder | O(n log n + m) | native GVE-Leiden, no post-ordering layer |
| 16 | `-o 16` | GoGraphOrder | O(m log d + n log n) | maximises forward-edge fraction |

## When to use what

| Workload | First try | Why |
|---|---|---|
| Power-law social graph + PR | `12:hrab` or `12:leiden` | best cache quality on community-strong graphs |
| Mesh / road network | `12:rcm` or `11:bnf` | RCM reduces bandwidth on sparse near-planar graphs |
| Iterative PR / SpMV with many trials | `12:hrab`, `12:leiden` | high quality amortises preprocessing cost |
| Single-pass BFS / SSSP from one source | HUBCLUSTERDBG (7) or `12:rabbit` | cheap reorder, hub-cached frontier |
| Dynamic / frequently-updated graph | `12:streaming` or RABBITORDER (8) | low memory + low reorder time |
| Quick experimentation, any graph | HUBCLUSTERDBG (7) | O(n), often within 10% of the best |
| Comparing against Gorder | GORDER (9) | the heavyweight cache-quality baseline |

## Algorithm details

### Baselines (0, 1)

**ORIGINAL** (`-o 0`) keeps the input ordering. Always run this first
to know what you're improving over.

**RANDOM** (`-o 1`) permutes vertices uniformly. Useful as a
worst-case reference and as a starting state for evaluating
deterministic reorderings on top of it.

### Degree-based (2-7)

All cheap (O(n) or O(n log n)) and effective on power-law graphs
where a small set of hubs dominates access frequency.

- **SORT** (`-o 2`): sort all vertices by degree, descending.
- **HUBSORT** (`-o 3`): same as SORT but framed around the hub idea.
- **HUBCLUSTER** (`-o 4`): split into hubs (high-degree) and non-hubs,
  reorder only the hubs, leave the rest in input order. Preserves
  non-hub spatial structure.
- **DBG** (`-o 5`): partition vertices into logarithmic degree
  buckets, place buckets contiguously. Hub bucket goes first.
- **HUBSORTDBG** (`-o 6`): DBG bucketing + sort by degree inside
  each bucket.
- **HUBCLUSTERDBG** (`-o 7`): DBG bucketing + hub-cluster inside
  each bucket. Reasonable default for power-law graphs when you
  want a cheap, predictable speedup.

### Community-based (8)

**RABBITORDER** (`-o 8`) — single-pass parallel Louvain that builds
a dendrogram of community merges, then orders vertices by DFS of that
dendrogram. Fast (~2-10× slower than degree-based, much faster than
Gorder) and produces high-quality cache locality on graphs with clear
community structure.

Variants:

| Flag | Implementation |
|---|---|
| `-o 8` or `-o 8:csr` | native CSR implementation (default) |
| `-o 8:boost` | original Boost-based implementation; requires Boost 1.58 |

The CSR variant has no Boost / numa / tcmalloc dependency and
benchmarks slightly faster.

### Heavyweight (9, 10)

**GORDER** (`-o 9`) — Wei et al. (2016). Sliding window of width 5
greedy vertex placement maximising a local cache-locality score
(Gscore). Produces best-in-class cache hits but is serial and
NP-hard in the limit; reorder time is typically 10-100× a community
method on the same graph. Use for paper comparisons.

**CORDER** (`-o 10`) — cache-aware bandwidth reduction. Less common;
included for completeness.

### Bandwidth-based (11)

**RCM** (`-o 11`) — Reverse Cuthill–McKee. BFS from a peripheral
vertex with neighbours visited in ascending-degree order, then
reverse the result. Bandwidth reduction translates directly to
sequential cache access on sparse, near-planar graphs (road
networks, finite-element meshes).

Variants:

| Flag | Description |
|---|---|
| `-o 11` | GoGraph-baseline RCM |
| `-o 11:bnf` | CSR-native George–Liu pseudoperipheral BFS |

### Composable (12 — the GraphBrew framework)

`-o 12` is the framework that combines community detection,
intra-community ordering, and inter-community arrangement into
ten variants. Variant selection is by flag suffix:

| Flag | Variant | One-line summary |
|---|---|---|
| `-o 12:leiden` (default) | Leiden | Leiden + BFS within community + hierarchical sort |
| `-o 12:rabbit` | Rabbit | single-pass Rabbit + dendrogram DFS |
| `-o 12:hrab` | HRAB | Leiden + RabbitOrder on super-graph + adaptive intra |
| `-o 12:tqr` | TQR | cache-line tile + RabbitOrder on tile adjacency |
| `-o 12:hcache` | HCache | uses every Leiden level as a cache-tier mapping |
| `-o 12:streaming` | Streaming | lazy-aggregation Leiden; lower memory |
| `-o 12:rcm` | RCM | Leiden + RCM within community + BNF-RCM super-graph |
| `-o 12:hubcluster` | HubCluster | Leiden + hub-first intra-community |
| `-o 12:rabbit:dbg` | Rabbit-DBG | single-pass Rabbit + DBG within community |
| `-o 12:rabbit:hubcluster` | Rabbit-HubCluster | single-pass Rabbit + hub-first intra |

Modifier tokens that compose with the variants:

| Token | Effect |
|---|---|
| `:sgres0.10` | super-graph modularity resolution γ (default 0.10) |
| `:gamma0.10` | alias for `:sgres` |
| `:rcm_intra` | force RCM within communities (HRAB default) |
| `:bfs_intra` | force BFS within communities |
| `:rcm_super` | RCM on super-graph instead of dendrogram DFS |
| `:hubx` | extract top-1% hubs first, place adjacent to their best block |
| `:gord` | Gorder-greedy intra-community (with UnitHeap) |

Example: `-o 12:hrab:sgres0.25:hubx`. See [GraphBrewOrder](GraphBrewOrder)
for the pipeline and how variants compose.

### Meta (13, 14)

**MAP** (`-o 13:<file>`) loads a vertex permutation from disk
(`.lo` or `.so` file). Used by the benchmark pipeline to apply a
pregenerated reordering without redoing the work.

**AdaptiveOrder** (`-o 14`) — runtime ML selector. Not part of the
VLDB submission; see [AdaptiveOrder-ML](AdaptiveOrder-ML).

### Reference Leiden (15)

**LeidenOrder** (`-o 15`) — direct GVE-Leiden ordering with no
GraphBrew post-processing layer. Use as a community-detection
baseline distinct from RabbitOrder's single-pass Louvain.

### Forward-edge maximisation (16)

**GoGraphOrder** (`-o 16`) — Zhou et al. (2024). Hub-aware BFS
followed by greedy vertex insertion that maximises the fraction of
edges where `src < dst` in the ordering. Specifically helps
Gauss-Seidel iterations (standard PR formulation). Has no effect
on double-buffered Jacobi algorithms like PR-SpMV.

Variants:

| Flag | Implementation |
|---|---|
| `-o 16` | default |
| `-o 16:fast` | parallel approximation |
| `-o 16:naive` | naive reference |

## Chained orderings

Multiple reorderings can be applied in sequence. Order matters:
community detection should precede degree refinement because the
degree-based methods preserve relative position within their buckets,
so they refine the existing community layout instead of destroying it.

```bash
# Leiden then DBG: community spatial layout + hub temporal locality
./bench/bin/pr -f g.el -s -o 12:leiden -o 5 -n 5

# HRAB then DBG
./bench/bin/pr -f g.el -s -o 12:hrab -o 5 -n 5

# Leiden then GoGraph: cache locality + convergence speed (PR only, not PR-SpMV)
./bench/bin/pr -f g.el -s -o 12:leiden -o 16 -n 5

# Rabbit then DBG (the lightweight chain)
./bench/bin/pr -f g.el -s -o 8 -o 5 -n 5
```

Five chains are formally evaluated in the VLDB paper §4.5.2.

## Algorithm selection cheatsheet

```
Is your graph mesh-like (road, FE)?
└── yes → 12:rcm   (or 11:bnf for standalone RCM)
└── no  → Does it have strong community structure?
          ├── yes (social, collaboration, web) → 12:hrab or 12:leiden
          └── no  (citation, sparse) → HUBCLUSTERDBG (7) or 12:rabbit
```

When in doubt, run HUBCLUSTERDBG (cheap baseline) and `12:hrab`
(usually best) and compare on your actual workload.

## Further reading

- [GraphBrewOrder](GraphBrewOrder) — the `-o 12` pipeline in detail
- [Cache-Simulation](Cache-Simulation) — measuring cache quality
- [Command-Line-Reference](Command-Line-Reference) — every flag
- [VLDB-Experiments](VLDB-Experiments) — paper reproduction
