# GraphBrewOrder

GraphBrewOrder (`-o 12`) builds a vertex permutation from three explicit
choices:

```text
partitioner -> block layout -> vertex layout
```

It changes vertex IDs and CSR placement, not graph topology.

[![GraphBrew architecture](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v4)

## Confirmed quality composition

```text
12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8
```

**Name:** LeidenGVE–SizeDesc–LocalGorder8

| Axis | Choice |
|---|---|
| Partitioner | GVE-Leiden path |
| Final supergraph order | none |
| Block layout | community size descending |
| Vertex layout | relaxed local Gorder, window 8 |

This fixed composition is the paper’s quality point. It beats each Rabbit
implementation and GORDER_csr in kernel geometric mean, maps faster than
GORDER_csr, and remains far more expensive to construct than Rabbit.
`LocalGorder8` distinguishes this relaxed per-community heuristic from the
faithful standalone `GORDER_csr` comparator.

## Compact-and-Emit

```text
12:leiden:compose:sg_none:comm_identity:
intra_bfs_compact_direct:cd_parallel:sgmb4096:norefine:1:1
```

[![Compact-and-Emit](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-compact-emit.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-compact-emit.svg?v=graphbrew-public-v4)

One-pass community detection can leave sparse representative labels.
Compact-and-Emit renumbers active labels monotonically and emits final IDs
during BFS. It removes empty-slot scheduling, the global local-ID array, and
the final sparse assignment sweep.

The resulting permutation is byte-identical to conventional BFS under the
validated controls. This is an implementation optimization, not a new
locality objective.

## Composition grammar

```text
12:<partitioner>:compose:<supergraph>:<community-order>:<intra-order>:...
```

Common explicit tokens:

| Stage | Tokens |
|---|---|
| Partitioner | `leiden`, `rabbit` |
| Supergraph order | `sg_none`, `sg_super_rabbit`, `sg_super_rcm`, `sg_hilbert` |
| Community order | `comm_identity`, `comm_size_desc`, `comm_size_asc`, `comm_degree_desc`, `comm_degree_asc` |
| Intra-community order | `intra_bfs`, `intra_rcm`, `intra_rcmpp`, `intra_gorder`, `intra_hubsort`, `intra_deg_asc` |
| Refinement | `refine_none`, `refine_2swap` |

Every published configuration pins all changed axes. Changing any token
creates a different treatment.

## Running example

The [GraphBrew Running Example](GraphBrew-Running-Example) follows one graph
through:

1. graph and CSR input;
2. community membership;
3. contiguous block placement;
4. local vertex ordering;
5. permutation validation and CSR relocation; and
6. the resulting property-access locality.

The small example demonstrates multiple local-layout options. It is not a
performance result and does not imply runtime composition search.

## Measurement contract

Report separately:

```text
mapping generation
permutation validation
CSR relocation
kernel-only time
mapping + reuse x kernel
```

Always include:

- ORIGINAL (`-o 0`);
- both Rabbit implementations when Rabbit is a comparator;
- exact graph and binary provenance;
- mapping fingerprints and draw policy;
- scheduler, nice value, affinity, governor, and turbo state;
- semantic verification and executed-work counters.

## Claim boundary

GraphBrew does not currently claim:

- one universal best composition;
- a Rabbit-cost-balanced arm that also beats ORIGINAL;
- a graph-held-out automatic generator; or
- a promoted CC-SV portal layout.

See [Evidence and Claims](Evidence-and-Claims).
