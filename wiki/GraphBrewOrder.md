# GraphBrewOrder

`-o 12` is the framework that produces all ten GraphBrew reordering
variants from a single three-stage pipeline.

```
Stage 1               Stage 2                Stage 3
community detection → intra-community order → inter-community arrangement
(Leiden / Rabbit)    (BFS, RCM, hub, DBG)   (hierarchy sort, Rabbit,
                                              RCM, tile, dendrogram)
```

The variant you ask for selects which algorithm runs at each stage.

## The variants

| Flag | Stage 1 | Stage 2 | Stage 3 | When to use |
|---|---|---|---|---|
| `-o 12:leiden` (default) | multi-pass Leiden | BFS in community | hierarchy sort | general purpose; balanced |
| `-o 12:rabbit` | single-pass Rabbit | dendrogram DFS | dendrogram order | lowest reorder cost; dynamic graphs |
| `-o 12:hrab` | multi-pass Leiden | adaptive BFS/RCM | Rabbit on super-graph | iterative workloads, dense graphs |
| `-o 12:tqr` | cache-line tiling | BFS within tile | Rabbit on tile graph | cache-geometry-sensitive workloads |
| `-o 12:hcache` | Leiden, all levels | BFS within finest | multi-level | deep hierarchical structure |
| `-o 12:streaming` | Leiden, lazy aggregation | BFS in community | hierarchy sort | memory-constrained / very large graphs |
| `-o 12:rcm` | multi-pass Leiden | RCM in community | BNF-RCM super-graph | sparse / mesh graphs (road networks) |
| `-o 12:hubcluster` | multi-pass Leiden | hub-first by degree | hierarchy sort | power-law graphs with dominant hubs |
| `-o 12:rabbit:dbg` | single-pass Rabbit | DBG bucketing | dendrogram order | fast reorder + degree refinement |
| `-o 12:rabbit:hubcluster` | single-pass Rabbit | hub-first by degree | dendrogram order | fast reorder + hub-aware |

All variants have $O(n\log n + m)$ time complexity. HRAB and TQR
additionally run RabbitOrder on a super-graph of size $n_c \ll n$;
that cost is dominated by Leiden on the original graph.

## The `compose` variant (pluggable axes)

`-o 12:compose` exposes the same three pipeline axes — super-graph
order, community order, intra-community order — as orthogonal CLI
picks.  Every other variant in the table above is a fixed configuration
of these axes; `compose` lets you mix them freely and was used to
prove that HRAB and TQR are literally compositions of the underlying
primitives (see `results/data/composability_phase6_final_2026_05_19.md`).

Three axes, two-or-more picks each:

| Axis | CLI prefix | Picks |
|---|---|---|
| Super-graph order (which communities sit next to which) | `sg_` | `none`, `super_rabbit`, `super_rcm`, `tile_rabbit` |
| Community order (sort key on top of the super-graph perm) | `comm_` | `size`, `degree`, `identity` |
| Intra-community order (vertex layout within a community) | `intra_` | `bfs`, `rcm` |

Examples:

```bash
# HRAB-equivalent
-o 12:compose:sg_super_rabbit:comm_identity:intra_rcm

# TQR-equivalent
-o 12:compose:sg_tile_rabbit:comm_identity:intra_bfs

# Pure intra (no super-graph), order communities by size, RCM inside
-o 12:compose:sg_none:comm_size:intra_rcm
```

Legacy aliases `s1_*`/`s2_*`/`s3_*` are still accepted (the older
parity sweeps and CI scripts use them); the new `sg_`/`comm_`/`intra_`
forms are the primary spelling and match the paper's vocabulary.

Defaults if any axis is omitted: `sg_none`, `comm_size`, `intra_bfs`.

## Modifier tokens

These compose with any variant after a `:`.

| Token | Effect | Default |
|---|---|---|
| `:sgres0.10` | super-graph modularity resolution γ in ΔQ = w − γ·str(u)·str(v)/(2·M) | 0.10 |
| `:gamma0.10` | alias for `:sgres` | — |
| `:rcm_intra` | force RCM within communities | on for `hrab`, off elsewhere |
| `:bfs_intra` | force BFS within communities | off |
| `:rcm_super` | RCM on super-graph instead of Rabbit dendrogram DFS | off |
| `:hubx` | extract top-1% hubs and place adjacent to their dominant block | off |
| `:gord` | Gorder-greedy intra-community via UnitHeap | off |
| `:norefine` | skip Leiden refinement phase | off (refine on) |

Example: `-o 12:hrab:sgres0.25:hubx` — HRAB variant with γ=0.25 and
hub extraction enabled.

## How the pipeline composes

For each variant, the pipeline (`bench/include/graphbrew/reorder/reorder_graphbrew.h`)
does:

1. **Detect communities** — Leiden multi-pass or Rabbit single-pass
   builds a `membership[v]` vector mapping each vertex to a community
   ID and a hierarchy / dendrogram describing the merge tree.
2. **Order within each community** — for every community `c`, build
   a local ordering `localIds[v]` of its members using BFS, RCM,
   hub-first sort, DBG, etc. (parallel per-community).
3. **Order the communities themselves** — produce a `commPerm[c]`
   permutation across communities. For `hrab` this comes from running
   RabbitOrder on the super-graph built from inter-community edge
   weights; for `leiden` it follows the Leiden hierarchy.
4. **Compose** — every vertex `v` lands at
   `newIds[v] = vertexOffsets[commPerm[membership[v]]] + localIds[v]`.

The composition keeps each community contiguous in memory (Stage 1 +
Stage 3 give the inter-community layout; Stage 2 gives the intra-community
layout) so that cache lines fetched for one vertex hold useful data
for its community neighbours.

## HRAB (the workhorse variant)

`-o 12:hrab` is the default we benchmark in the paper. Pipeline:

1. Multi-pass Leiden produces ~100K communities on a typical 100M-edge
   social graph.
2. Build a super-graph where each community is a vertex and edge
   weights aggregate the inter-community connectivity.
3. Run RabbitOrder on the super-graph with modularity gain
   ΔQ = w − γ·str(u)·str(v)/(2·M_super) and γ = `:sgres` (default 0.10).
   This merges Leiden's fine communities into ~1-5K cache-sized blocks
   and assigns them a dendrogram-DFS order.
4. Within each surviving block, apply intra-community ordering.
   The default is **adaptive**: blocks with > 4096 vertices use BFS
   (better for dense graphs with huge natural communities), smaller
   blocks use the full BNF-RCM pipeline (better for sparse graphs).
   This adaptive split was empirically tuned in May 2026 — see
   `results/data/empirical_validation_FINAL_2026_05_18.md`.

For why each step matters, see the paper §3.3.2.

## When to override the defaults

| Symptom | Try |
|---|---|
| HRAB reorder takes too long on a 1B-edge graph | `-o 12:rabbit` (skip Leiden entirely) or `-o 12:streaming` (lazy aggregation) |
| HRAB cache quality is great but real kernel is slower than ORIGINAL | force `:bfs_intra` (already adaptive default, but you can pin it) |
| Memory pressure during build | `-o 12:streaming` cuts aggregation peak to O(n) |
| Need to compare against a specific γ | `-o 12:hrab:sgres0.05` (more aggressive) or `:sgres1.0` (faithful Rabbit) |
| Sparse / mesh graph | `-o 12:rcm` — Leiden first then RCM per community |

## Implementation files

| Function | File | Purpose |
|---|---|---|
| `GenerateGraphBrewMappingUnified` | `bench/include/external/gapbs/builder.h` | top-level dispatch |
| `parseGraphBrewConfig` | `reorder_graphbrew.h` | turns `12:tokens:…` into a `GraphBrewConfig` |
| `orderHybridLeidenRabbit` | `reorder_graphbrew.h` (~L3444) | HRAB; the most heavily-commented implementation |
| `orderTileQuantizedRabbit` | `reorder_graphbrew.h` (~L4920) | TQR |
| `orderHierarchicalCacheAware` | `reorder_graphbrew.h` (~L3035) | HCache |
| `CommunityScanner` | `reorder_graphbrew.h` (~L601) | sparse open-address hashmap used by all variants' super-graph build |

## Chaining

GraphBrew variants compose with later passes via repeated `-o` flags:

```bash
# Leiden communities, then DBG refinement
./bench/bin/pr -f g.el -s -o 12:leiden -o 5 -n 5

# HRAB then GoGraph (forward-edge maximisation for PR Gauss-Seidel)
./bench/bin/pr -f g.el -s -o 12:hrab -o 16 -n 5
```

See [Reordering-Algorithms#chained-orderings](Reordering-Algorithms)
for the five chains evaluated in the paper.

## Output

Running with `-o 12:hrab` prints a community-size histogram and the
final number of super-communities:

```
  hybrid-rabbit: 99489 Leiden communities
  hybrid-rabbit: super-graph M=56371200
  hybrid-rabbit: 48266 super-communities (merged from 99489)
  comm-sizes: <=3: 25653 comms | 4-10: 24012 | … | >10K: 10 | max=98888
  hybrid-rabbit-rcm-intra: tiny=2 small=60479 med=2563 large=30
Reorder Time:        5.09693
```

This is the canonical signal that the pipeline ran correctly. The
`max=` in `comm-sizes` is the largest single community — if it's >5%
of N on a sparse graph, something went wrong (Leiden likely collapsed
the graph into one mega-community; check the input is connected).

## Further reading

- [Reordering-Algorithms](Reordering-Algorithms) — every algorithm including the non-GraphBrew baselines
- [Cache-Simulation](Cache-Simulation) — how to measure cache quality of a variant
- [Code-Architecture](Code-Architecture) — codebase map
- [VLDB-Experiments](VLDB-Experiments) — the paper's variant evaluation
