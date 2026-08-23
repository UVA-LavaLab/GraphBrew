# Reordering Algorithms

GraphBrew implements 17 algorithm IDs (`-o 0` through `-o 16`).
Two are baselines (no useful reordering), one is a runtime selector
(AdaptiveOrder), one loads a precomputed permutation
from disk (MAP). The remaining 13 produce orderings you can benchmark.

## Why reorder

Power-law and clustered graphs have neighbour-access patterns that
miss in cache when vertices are placed randomly. Renumbering vertices
so that frequently co-accessed ones land on nearby cache lines turns
unpredictable misses into hits. Reordering does **not** change the
graph's topology — only the integer labels.

Three interacting locality dimensions matter:

| Dimension | Captured by | Example algorithms |
|---|---|---|
| Spatial (community structure) | Leiden, Rabbit Order | `12:leiden`, `12:rabbit`, RABBIT (8) |
| Temporal (degree skew) | hub grouping | HUBCLUSTER (4), DBG (5) |
| Directed convergence (separate extension) | edge-direction optimisation | GoGraph (16), chained `12:leiden → 16`; not a claim on the symmetric main corpus |

GraphBrew (`-o 12`) is the framework that composes these dimensions;
the other IDs are individual primitives or baselines.

## One shared comparison input

[![Shared reordering input](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/example-input.svg?v=graphbrew-public-v3)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/example-input.svg?v=graphbrew-public-v3)

Every catalog strip uses this exact nine-vertex graph and records the converter
output order under one pinned binary. That makes equal outputs visible and
prevents algorithm-specific figures from quietly changing the topology.

## Quick reference

See the [Reordering Figure Catalog](Reordering-Figure-Catalog) for a
measured output strip, mechanism-specific explanation, and generated
draw.io source for every algorithm ID.

| ID | Flag | Algorithm | Complexity | Notes |
|---|---|---|---|---|
| 0 | `-o 0` | ORIGINAL | O(1) | input ordering, baseline |
| 1 | `-o 1` | RANDOM | O(n) | thread-independent SplitMix64 shuffle, seed 0 |
| 2 | `-o 2` | SORT | O(n log n) | degree-descending sort |
| 3 | `-o 3` | HUBSORT | O(n log n) | sorted hubs; preserve non-hub source IDs when possible |
| 4 | `-o 4` | HUBCLUSTER | O(n) | stable hubs; preserve non-hub source IDs when possible |
| 5 | `-o 5` | DBG | O(n) | logarithmic degree buckets |
| 6 | `-o 6` | HUBSORTDBG | O(n log n) | compact two-bucket DBG with sorted hubs |
| 7 | `-o 7` | HUBCLUSTERDBG | O(n) | compact two-bucket DBG with stable hubs |
| 8 | `-o 8` | RABBITORDER | O(n log n + m) | Louvain + dendrogram DFS; variants `csr` (default), `boost` |
| 9 | `-o 9:csr` | GORDER | O(n·w + m) | faithful CSR sliding-window greedy, w=5; `9:gograph` forces the legacy validation path and bare `9` auto-selects CSR above the 32-bit edge range |
| 10 | `-o 10` | CORDER | O(n) | hot/cold workload segments; `10` historical 1K, `10:canonical` upstream 1 MiB |
| 11 | `-o 11` | RCM | O(n log n + m) | historical double-pass; variants `mind`, `bnf` expose explicit single-pass methods |
| 12 | `-o 12` | GraphBrewOrder | O(n log n + m) | composable pipeline — see [GraphBrewOrder](GraphBrewOrder) |
| 13 | `-o 13:<file>` | MAP | O(n) | load permutation from `.lo` / `.so` file |
| 14 | `-o 14` | AdaptiveOrder | varies | validated deterministic rule plus legacy offline-model modes; see [AdaptiveOrder](AdaptiveOrder) |
| 15 | `-o 15` | LeidenOrder | O(n log n + m) | GVE-Leiden communities plus an explicit post-layout policy |
| 16 | `-o 16` | GoGraphOrder | O(m log d + n log n) | M-maximizing core diagnostic; published Rabbit clustering omitted |

## Evidence-scoped selection

Select an objective before selecting an ordering. These rows summarize frozen
measurements, not universal graph-type rules. Exact evidence and confidence
intervals are recorded in
[`docs/recommendation-evidence.json`](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/recommendation-evidence.json).

| Objective | First measured configuration | What is proved |
|---|---|---|
| Non-Rabbit all-kernel ordering quality | `12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8` | Kernel-only GM is 1.049x over Rabbit CSR and 1.059x over Boost across 11 graphs and 7 kernels; both lower 95% bounds exceed one. Full Leiden mapping costs about 17–18x Rabbit, and a one-pass replacement failed to retain the quality win. |
| Non-Rabbit controlled-work quality | `12:leiden:compose:sg_none:comm_size_desc:intra_rcmpp` | About 1.025x over both Rabbit implementations for fixed-work PR/PR-SpMV, fixed-source SSSP, and BC. It is not the all-kernel winner because CC/CC-SV regress. |
| Low-reuse end-to-end | `12:leiden:compose:sg_none:comm_identity:intra_gorder:gw32:gordf500:cd_parallel:1:1` | Wins reuse-1 on a bounded rapid cohort through lower mapping cost; Rabbit kernels are faster on average, and Boost wins the Friendster holdout. |
| Automatic all-kernel reuse 1–2 | `14:_:_:_:allkernel-lowreuse-rule:best-endtoend:<reuse>` | Frozen deterministic rule selects the cost-matched FastLeiden-Gorder8 composition or Boost Rabbit. Seven selected untouched graphs pass at 1.696x/1.642x over Boost; including five Rabbit fallbacks, the frozen portfolio reaches 1.361x/1.336x over always-Boost. Public portfolio accounting does not store algorithm-14 feature time. |
| Road/mesh diagnosis | Compare `12:hrab`, `12:hrab:bfs_intra`, `11:bnf`, and both Rabbits | HRAB-RCM has the best point estimate on the single road and mesh graphs, but each type has one graph, so this is descriptive rather than a general recommendation. |
| Unknown workload | `0`, `8:csr`, `8:boost`, plus one objective-matched COMPOSE row | Measure kernel-only quality and end-to-end time separately. Graph type alone does not identify the winner. |

## Algorithm details

### Baselines (0, 1)

**ORIGINAL** (`-o 0`) keeps the input ordering. Always run this first
to know what you're improving over.

**RANDOM** (`-o 1`) uses a specified seed-0 SplitMix64/Fisher-Yates
permutation. It is a controlled shuffled labeling, not a worst-case claim.

### Degree-based (2-7)

All are cheap (O(n) or O(n log n)) degree-layout controls. Their benefit is
graph- and kernel-dependent even when a small set of hubs dominates access.

- **SORT** (`-o 2`): sort all vertices by degree, descending.
- **HUBSORT** (`-o 3`): sort only vertices above average degree, then
  preserve non-hub source IDs whenever the permutation permits.
- **HUBCLUSTER** (`-o 4`): split into hubs (high-degree) and non-hubs,
  reorder only the hubs, leave the rest in input order. Preserves
  non-hub spatial structure.
- **DBG** (`-o 5`): partition vertices into logarithmic degree
  buckets, place buckets contiguously. Hub bucket goes first.
- **HUBSORTDBG** (`-o 6`): compact hubs first, sorted by degree; compact
  non-hubs after them.
- **HUBCLUSTERDBG** (`-o 7`): compact stable hubs first and stable non-hubs
  second. Use it as a cheap comparison point, not as a guaranteed default.

### Community-based (8)

**RABBITORDER** (`-o 8`) — single-pass parallel incremental aggregation that builds
a dendrogram of community merges, then orders vertices by DFS of that
dendrogram. Fast (~2-10× slower than degree-based, much faster than
Gorder) and produces high-quality cache locality on graphs with clear
community structure. Standalone Rabbit mappings are schedule-sensitive;
GraphBrew records a stable permutation fingerprint for every draw, and final
studies use explicitly versioned repeated draws rather than cherry-picking.

Variants:

| Flag | Implementation |
|---|---|
| `-o 8` or `-o 8:csr` | native CSR implementation (default) |
| `-o 8:boost` | original Boost-based implementation; requires Boost 1.58 |

The CSR variant has no Boost / numa / tcmalloc dependency. Relative speed is
graph-dependent: neither implementation is a universal winner.

### Heavyweight (9, 10)

**GORDER** (`-o 9:csr` for paper runs) — Wei et al. (2016). Sliding window of width 5
greedy vertex placement maximising a local cache-locality score
(Gscore). Targets a strong window-locality objective but is serial and
NP-hard in the limit; its measured reorder time is often much larger than a
community method on the same graph. `-o 9:gograph` forces the mapping-equivalent
legacy validation path; bare `-o 9` is compatibility auto mode. `-o 9:fast`
is a distinct relaxed mapping with fixed batch/window semantics and explicit
environment overrides.

**CORDER** (`-o 10`) — degree-based hot/cold workload balancing. Bare
`-o 10` preserves GraphBrew's historical 1,024-vertex partitions;
`-o 10:canonical` uses the upstream 1 MiB float-property segment.

### Bandwidth-based (11)

**RCM** — bandwidth-oriented BFS ordering for sparse, near-planar graphs.
Bare `-o 11` is retained only for historical compatibility: it applies a
MIND-start RCM, rebuilds the graph, then applies a second RCM. Use an explicit
single-pass variant for new comparisons.

Variants:

| Flag | Description |
|---|---|
| `-o 11` | Historical double-pass MIND composition |
| `-o 11:mind` | Single-pass GoGraph MIND-start RCM |
| `-o 11:bnf` | CSR-native George–Liu/BNF pseudoperipheral RCM |

### Composable (12 — the GraphBrew framework)

`-o 12` explicitly composes a community detector, community-block layout,
and intra-community layout. These parameters are hand selected; GraphBrew
does not search them at runtime.

Two evidence-bound examples are:

| Objective | Exact configuration |
|---|---|
| Kernel-only all-kernel quality | `12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8` |
| Cost-matched low-reuse candidate | `12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8:cd_parallel:sgmb4096:gordf5000:norefine:2:2` |

Named historical presets such as `hrab`, `tqr`, and `hcache` remain callable
for reproduction. They are not aliases for the promoted composition and
should not be treated as automatic recommendations. See
[GraphBrewOrder](GraphBrewOrder) for stage tokens and the measurement
contract.

The six rejected novelty directions remain only as removable correctness
prototypes under `reorder/experimental/`; none is a public algorithm or CLI
variant. The evidence-bound `intra_gorder` token remains the historical
relaxed direct-neighbor heuristic. See
[GraphBrewOrder](GraphBrewOrder#contained-validity-only-prototypes).

### Meta (13, 14)

**MAP** (`-o 13:<file>`) loads a vertex permutation from disk
(`.lo` or `.so` file). Used by the benchmark pipeline to apply a
pregenerated reordering without redoing the work.

**AdaptiveOrder** (`-o 14`) — runtime selection boundary. The validated path
is the frozen deterministic `allkernel-lowreuse-rule`, not ML. Historical
offline-model modes remain for compatibility; see
[AdaptiveOrder](AdaptiveOrder).

### Reference Leiden (15)

**LeidenOrder** (`-o 15`) — GVE-Leiden community detection followed by a
GraphBrew-defined vertex layout. Bare/numeric forms preserve the historical
`hierarchy-degree` layout. The full syntax is
`15:<resolution>:<iterations>:<passes>:<layout>`, where layout is
`hierarchy-degree`, `final-stable`, or `final-degree`. This is not a native
ordering defined by the Leiden paper; use the explicit layouts as controlled
community-to-ordering policies.

### Forward-edge maximisation (16)

**GoGraphOrder** (`-o 16`) — core of Zhou et al. (ICDE 2024).
Hub-aware BFS is followed by greedy insertion that maximizes edges where
`src < dst`. The published pipeline first applies RabbitOrder clustering and
orders the cluster graph; GraphBrew currently omits that stage, so Algorithm
16 is a diagnostic rather than a faithful standalone baseline. On symmetric
graphs the M objective is constant. It targets asynchronous/Gauss-Seidel
convergence, not double-buffered Jacobi kernels such as PR-SpMV.

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

Chains are also explicit, hand-configured treatments. Record the complete
sequence and measure its combined mapping cost.

## Selection checklist

1. Run `-o 0` to establish the current-layout baseline.
2. Decide whether the objective is kernel quality or low-reuse time-to-solution.
3. Include both `8:csr` and `8:boost`; they can reverse order by graph.
4. Add the exact COMPOSE row from the evidence-scoped table that matches the
   objective.
5. Use pre-generated mappings so every kernel sees the same permutation.
6. Report mapping and kernel time separately before reporting amortized totals.

## Further reading

- [GraphBrewOrder](GraphBrewOrder) — the `-o 12` pipeline in detail
- [Cache-Simulation](Cache-Simulation) — measuring cache quality
- [Command-Line-Reference](Command-Line-Reference) — every flag
- [Reproducible-Experiments](Reproducible-Experiments) — frozen study reproduction
