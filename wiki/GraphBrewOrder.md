# GraphBrewOrder

GraphBrewOrder (`-o 12`) exposes vertex reordering as a composition of
independent stages rather than one opaque algorithm name.

## Composition model

For a graph \(G=(V,E)\), GraphBrew constructs a permutation in four steps:

1. detect communities \(C_0,\ldots,C_{k-1}\);
2. optionally derive a permutation of the community super-graph;
3. place the community blocks;
4. order vertices inside each block.

The three public composition axes are:

| Axis | Question |
|---|---|
| Super-graph order | Should connectivity between communities determine their coarse placement? |
| Community order | How should the community blocks be arranged? |
| Intra-community order | How should vertices be arranged within each block? |

The community detector is an upstream choice. Current experiments use Leiden
or Rabbit.

## Explicit means hand configured

An algorithm-12 string executes the supplied stages:

```bash
./bench/bin/pr -f graph.sg -s \
  -o '12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8' \
  -n 3
```

GraphBrew does not search the composition space at runtime. Every token is
part of the experimental treatment and should be recorded with the result.
Automatic deployment is a separate algorithm-14 policy; see
[AdaptiveOrder](AdaptiveOrder).

## Core stage tokens

### Community detector

| Token | Meaning |
|---|---|
| `leiden` | GVE-Leiden CSR aggregation |
| `cd_parallel` | parallel community-detection schedule |
| `cd_serial` | deterministic serial schedule |
| `rabbit` | Rabbit community detection instead of Leiden |

### Super-graph order

| Token | Meaning |
|---|---|
| `sg_none` | no separate super-graph permutation |
| `sg_super_rabbit` | Rabbit on the community super-graph |
| `sg_super_rcm` | RCM on the community super-graph |
| `sg_tile_rabbit` | tile-quantized Rabbit |
| `sg_hilbert` | Hilbert order of community size and average degree |

### Community block order

| Token | Meaning |
|---|---|
| `comm_identity` | retain the preceding community permutation |
| `comm_size_desc`, `comm_size_asc` | sort by community size |
| `comm_degree_desc`, `comm_degree_asc` | sort by total community degree |
| `comm_cut_min` | crossing-edge nearest-neighbor heuristic |

### Intra-community order

| Token | Meaning |
|---|---|
| `intra_bfs` | BFS from a high-degree seed |
| `intra_rcm`, `intra_rcmpp` | reverse Cuthill-McKee variants |
| `intra_gorder` | Gorder within each community |
| `intra_hubsort` | descending degree |
| `intra_deg_asc` | ascending degree control |
| `intra_boundary_last` | interior vertices before boundary vertices |
| `intra_core` | descending core number |
| `intra_random` | deterministic random control |
| `intra_dendrogram` | Rabbit dendrogram DFS; Rabbit detector only |

Additional experimental primitives remain callable in the parser, but are not
public recommendations.

## Cost controls

| Token | Meaning |
|---|---|
| `sgmb<N>` | ordered super-graph proposal batch size |
| `gw<N>` | Gorder window |
| `gordf<N>` | use BFS instead of Gorder for communities larger than `N` |
| `norefine` | disable Leiden refinement |
| terminal integers | maximum Leiden iterations, then maximum passes |

`sgmb<N>` computes proposals in parallel within each batch and commits the
batch in community order. `sgmb1` preserves sequential commit semantics.

## Promoted cost-matched composition

```text
12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8:
cd_parallel:sgmb4096:gordf5000:norefine:2:2
```

This configuration combines:

- two parallel Leiden iterations and passes;
- ordered proposal batches of 4096;
- SizeDesc block placement;
- Gorder8 for communities with at most 5000 vertices;
- BFS fallback for larger communities; and
- no refinement.

It is the GraphBrew branch used by the validated low-reuse selector. On its
five-graph cost-matched confirmation set, it was cheaper to construct and
faster across the seven-kernel aggregate than both Rabbit implementations.
That result does not make it universal: road/mesh graphs and some large graphs
showed why the selector still needs a Rabbit fallback.

## Reading a composition string

```text
12 : leiden : compose : sg_none : comm_size_desc : intra_gorder
   : gw8 : cd_parallel : sgmb4096 : gordf5000 : norefine : 2 : 2
```

| Field | Decision |
|---|---|
| `12` | GraphBrewOrder dispatcher |
| `leiden` | Leiden aggregation |
| `compose` | pluggable stage pipeline |
| `sg_none` | no coarse super-graph reorder |
| `comm_size_desc` | largest community blocks first |
| `intra_gorder`, `gw8` | local Gorder with window 8 |
| `cd_parallel` | parallel community detection |
| `sgmb4096` | batched ordered super-graph moves |
| `gordf5000` | BFS fallback above 5000 vertices |
| `norefine` | omit refinement |
| `2:2` | iteration and pass budgets |

## Measurement contract

A composition must be evaluated with both:

```text
kernel-only time
mapping time + reuse * kernel time
```

Always compare against:

- ORIGINAL (`-o 0`);
- CSR Rabbit (`-o 8:csr`); and
- Boost Rabbit (`-o 8:boost`) when available.

Report the exact composition, graph provenance, build identity, threads,
affinity, trials, and reuse. Reuse counts complete kernel invocations sharing
one materialized mapping.

## Related pages

- [All-Kernel Low-Reuse Selector](All-Kernel-Low-Reuse-Selector)
- [Reordering Algorithms](Reordering-Algorithms)
- [Reproducible Experiments](Reproducible-Experiments)
- [Command-Line Reference](Command-Line-Reference)
