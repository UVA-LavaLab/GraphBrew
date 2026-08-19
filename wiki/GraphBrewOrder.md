# GraphBrewOrder

`-o 12` is the framework that produces the registered GraphBrew reordering
variants from one configurable pipeline.

```
Partitioner          Block layout             Vertex layout
Leiden / Rabbit  ->  community arrangement -> BFS / RCM / Gorder / degree
```

These decisions interact; COMPOSE makes each one explicit so it can be
measured without hiding multiple changes behind one variant name.

## The variants

| Flag | Partitioner | Block layout | Vertex layout | Evidence status |
|---|---|---|---|---|
| `-o 12:leiden` (default) | GVE-Leiden | size-desc blocks | per-block Rabbit | Highest controlled-work point estimate in the core matrix; expensive preprocessing. |
| `-o 12:rabbit` | Rabbit | Rabbit dendrogram | dendrogram DFS | Strong low-to-medium-reuse core result; schedule-sensitive. |
| `-o 12:hrab` | Leiden | Rabbit super-graph | RCM | Best road/mesh point estimate, but each type has one graph. |
| `-o 12:hrab:bfs_intra` | Leiden | Rabbit super-graph | BFS | Controlled intra-order comparison for HRAB. |
| `-o 12:tqr` | Leiden | Rabbit-ordered tiles | BFS | Measured core variant; no universal cache-geometry recommendation is established. |
| `-o 12:hcache` | Leiden hierarchy | hierarchy levels | BFS | Measured hierarchy diagnostic, not a cache-fit guarantee. |
| `-o 12:streaming` | lazy Leiden | size-desc blocks | per-block Rabbit | Measured aggregation variant; dynamic-update and peak-memory superiority are not established. |
| `-o 12:hubcluster` | Leiden | global hub split | degree-desc | Measured hub-layout diagnostic. |
| `-o 12:rabbit:dbg` | Rabbit | community blocks | DBG | Lower-cost Rabbit composition; report mapping and kernel effects separately. |
| `-o 12:rabbit:hubcluster` | Rabbit | global hub split | degree-desc | Measured Rabbit hub-layout diagnostic. |

Actual cost depends on the selected primitives, number of Leiden passes,
community sizes, and any serial intra-community work. Use measured complete
mapping time rather than one complexity label for every composition.

## The `compose` variant (pluggable axes)

`-o 12:compose` exposes the same three pipeline axes — super-graph
order, community order, intra-community order — as explicit CLI picks.
Every other variant in the table above is a fixed configuration of these
interacting axes. `compose` lets experiments change one registered field at a
time and validate the structured effective and realized configurations.

Three axes, two-or-more picks each:

| Axis | CLI prefix | Picks |
|---|---|---|
| Super-graph order (which communities sit next to which) | `sg_` | `none`, `super_rabbit`, `super_rcm`, `tile_rabbit`, `hilbert` |
| Community order (sort key on top of the super-graph perm) | `comm_` | `size`, `size_asc`, `degree_desc`, `degree_asc`, `identity`, `cut_min` |
| Intra-community order (vertex layout within a community) | `intra_` | `bfs`, `rcm`, `rcmpp`, `hubsort`, `hub2`, `deg_asc`, `alternate`, `random`, `bndlast`, `core`, `dendrogram`, `gorder` |
| Refinement pass (post-intra polish) | `refine_` | `none`, `2swap` (adjacent-swap FM polish) |

Intra-community picks at a glance:

- `bfs` / `rcm` — original BFS-from-hub and reverse Cuthill–McKee.
- `rcmpp` (alias `rcm++`) — RCM++ (Hou/Liu/Zhu, arXiv 2409.04171, 2024). Same CM-BFS body as `rcm` but the initial start vertex is picked by the bi-criteria score `argmin(0.5·deg_rank + 0.5·depth_rank)` instead of plain BNF min-degree. Costs one extra per-community BFS; sometimes finds a more peripheral seed in a single shot. Skipped for `sz ≤ 32` (overhead dominates).
- `hubsort` (alias `hub`) — sort by degree desc inside the community. Cheap
  (no traversal); performance is graph- and kernel-dependent.
- `deg_asc` — sort by degree asc. Disperses hubs to high IDs; reduces false-sharing on `visited[]`/`father[]` CAS lines for parallel BFS.
- `alternate` (alias `alt`) — interleave `[hub, leaf, hub, leaf, ...]` after a degree-desc sort.
- `random` — seeded shuffle, useful as a sanity-check baseline.
- `bndlast` (alias `boundary_last`) — community-internal vertices first, cross-community boundary vertices last.
- `core` — k-core order: peel by minimum degree, place the deepest core at the highest IDs.
- `dendrogram` — DFS over the Rabbit dendrogram (no extra traversal; reuses the merge tree).
- `gorder` — Gorder window-greedy via the per-community subgraph; pair with
  `gw<N>` to set the window. The measured `gw8` SizeDesc composition beats
  both Rabbit implementations in the eleven-graph all-kernel aggregate, but
  requires the full multi-pass Leiden partition and has high preprocessing
  cost. The measured one-pass replacement is cheaper than Rabbit but loses
  kernel quality.

Examples:

```bash
# COMPOSE control using a Rabbit super-graph and RCM intra order
-o 12:compose:sg_super_rabbit:comm_identity:intra_rcm

# COMPOSE control using Rabbit-ordered tiles and BFS intra order
-o 12:compose:sg_tile_rabbit:comm_identity:intra_bfs

# Pure intra (no super-graph), order communities by size, RCM inside
-o 12:compose:sg_none:comm_size:intra_rcm

# Leiden + per-community hub-first sort, communities ordered by total degree desc
-o 12:leiden:compose:comm_degree_desc:intra_hubsort

# 4-axis: Rabbit super-graph × degree-desc community order × hub-first intra
-o 12:rabbit:compose:sg_super_rabbit:comm_degree_desc:intra_hubsort

# Leiden + Gorder with a wider window than the default 5
-o 12:leiden:compose:intra_gorder:gw8
```

Legacy aliases `s1_*`/`s2_*`/`s3_*` are still accepted (the older
parity sweeps and CI scripts use them); the new `sg_`/`comm_`/`intra_`/`refine_`
forms are the primary spelling and match the paper's vocabulary.

Defaults if any axis is omitted: `sg_none`, `comm_size`, `intra_bfs`, `refine_none`.

## Modifier tokens

These compose with any variant after a `:`.

| Token | Effect | Default |
|---|---|---|
| `:sgres0.10` | super-graph modularity resolution γ in ΔQ = w − γ·str(u)·str(v)/(2·M) | 0.10 |
| `:gamma0.10` | alias for `:sgres` | — |
| `:gw<N>` | Gorder window size (only meaningful with `intra_gorder`) | 5 |
| `:cd_rabbit` / `:cd_leiden` | force the community-detection backend after a preset (`12:`, `9:`) | preset's CD |
| `:rcm_intra` | force RCM within communities | on for `hrab`, off elsewhere |
| `:bfs_intra` | force BFS within communities | off |
| `:rcm_super` | RCM on super-graph instead of Rabbit dendrogram DFS | off |
| `:hubx` | extract the configured hub fraction (default 0.1%) and place it adjacent to the dominant block | off |
| `:gord` | Gorder-greedy intra-community via UnitHeap | off |
| `:refine_2swap` | adjacent-swap FM polish after intra-community ordering | off |
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

## HRAB

`-o 12:hrab` is one measured core variant. Pipeline:

1. Multi-pass Leiden produces ~100K communities on a typical 100M-edge
   social graph.
2. Build a super-graph where each community is a vertex and edge
   weights aggregate the inter-community connectivity.
3. Run RabbitOrder on the super-graph with modularity gain
   ΔQ = w − γ·str(u)·str(v)/(2·M_super) and γ = `:sgres` (default 0.10).
   This merges Leiden's fine communities into ~1-5K cache-sized blocks
   and assigns them a dendrogram-DFS order.
4. Within each surviving block, apply RCM by default; use
   `:bfs_intra` for the controlled BFS counterpart.

The super-graph axis was harmful on average in its registered one-axis
contrast, even though HRAB has the best point estimate on the single road and
mesh graphs. Treat those topology rows as descriptive, not universal guidance.

## Evidence-based diagnostics

| Question | Compare |
|---|---|
| Is super-graph ordering useful? | `sg_none` versus one explicit `sg_*` value with the same `comm_*` and `intra_*`. |
| Does intra Gorder improve quality enough to amortize? | `intra_bfs` versus `intra_gorder:gw<N>` with identical partition and block axes. |
| Is a block sort helping? | `comm_identity` versus one explicit sort while holding partition and intra order fixed. |
| Is a fast mapping preferable to Rabbit? | Report kernel-only ratios and `map + reuse * kernel` against both `8:csr` and `8:boost`. |
| Is road/mesh behavior general? | Add more than one graph of that type before making a type-level claim. |

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

This output confirms which pipeline ran. Community count or maximum size is a
diagnostic, not by itself a correctness or quality test.

## Further reading

- [Reordering-Algorithms](Reordering-Algorithms) — every algorithm including the non-GraphBrew baselines
- [Cache-Simulation](Cache-Simulation) — how to measure cache quality of a variant
- [Code-Architecture](Code-Architecture) — codebase map
- [Reproducible-Experiments](Reproducible-Experiments) — frozen variant evaluation

## Compose tokens added 2026-05-21 (8h autonomous run)

| Token | Axis | Mechanism | Status |
|---|---|---|---|
| `intra_hub2` | IntraCommunityOrder | Sort by Σ deg(neighbor) descending (DRO/Lakhotia IISWC'19) | Callable; **negative result**: loses 0/12 vs §19 champions — see v5 §40.2 |
| `comm_cut_min` | CommunityOrder | NN-TSP over inter-community crossing-edge graph (Mt-METIS LaSalle IPDPS'13). Falls back to DegreeDesc if C>4096. | Callable; **wins 2/12 cells**: cit-P CC −8.1%, pokec PR −4.4% vs §19 champions — see v5 §40 |
| `sg_hilbert` | SuperGraphOrder | 2-D 8-bit Hilbert curve over (community size, avg degree) (Mosaic EuroSys'17) | Callable; **wins 1/12 cells marginally**: cit-P BFS −1.1% (within n=5 σ; needs n=10 hardening) — see v5 §40 |

All three slot into the existing parser with zero CLI changes. Composable
with any other axis. Reference recipe:

```
-o 12:cd_leiden:compose:comm_degree_desc:intra_hub2
-o 12:cd_leiden:compose:comm_cut_min:intra_hubsort
-o 12:cd_leiden:compose:sg_hilbert:comm_identity:intra_hubsort
```
