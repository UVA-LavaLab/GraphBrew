# GraphBrewOrder

GraphBrewOrder (`-o 12`) builds a vertex permutation from three explicit
choices:

```text
partitioner -> block layout -> vertex layout
```

It changes vertex IDs and CSR placement, not graph topology.

[![GraphBrew architecture](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v4)

## Layout expression

Let `P(v)` assign vertex `v` to a block, `B` order the blocks, and `L[c](v)`
give the local position of `v` inside block `c`. GraphBrew emits:

```text
pi(v) = block_offset(B(P(v))) + L[P(v)](v)
```

If `B` and each local `L[c]` are permutations, `pi` is a permutation because
the block intervals are disjoint and cover every final ID. Requested and
realized expressions are recorded separately so a fallback cannot silently
change the treatment.

## Composition grammar

```text
12:<partitioner>:compose:<supergraph>:<community-order>:<intra-order>:...
```

Common explicit tokens:

| Stage | Tokens |
|---|---|
| Partitioner | `leiden`, `rabbit` |
| Supergraph order | `sg_none`, `sg_super_rabbit`, `sg_super_rcm`, `sg_hilbert` |
| Block order | `comm_identity`, `comm_size_desc`, `comm_size_asc`, `comm_degree_desc`, `comm_degree_asc` |
| Intra-block order | `intra_bfs`, `intra_rcm`, `intra_rcmpp`, `intra_gorder`, `intra_hubsort`, `intra_deg_asc` |
| Refinement | `refine_none`, `refine_2swap` |

Changing any token creates a different layout configuration.

Example:

```text
12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8
```

This means: use the Leiden partitioner, do not reorder the quotient graph,
place larger blocks first, and apply the local Gorder heuristic with window
eight inside each block. It is an explicit example, not a universal default.

## Compact-and-Emit

```text
12:leiden:compose:sg_none:comm_identity:
intra_bfs_compact_direct:cd_parallel:sgmb4096:norefine:1:1
```

[![Compact-and-Emit](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-compact-emit.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-compact-emit.svg?v=graphbrew-public-v4)

One-pass community detection can leave sparse representative labels.
Compact-and-Emit renumbers active labels monotonically and writes final IDs
during BFS. It removes empty-slot scheduling, the global local-ID array, and
the final sparse assignment sweep.

Use permutation fingerprints and semantic verification to confirm that the
optimized path produces the intended layout.

## How stages can affect execution

Different stages target different structural effects:

| Stage | Typical effect to measure |
|---|---|
| Partitioner | block membership and cross-block edges |
| Block order | global placement of groups and ID-sensitive traversal behavior |
| Intra-block order | local co-access distance, bandwidth, or hub placement |
| Refinement | additional locality improvement versus construction cost |

These are hypotheses, not guarantees. Compare the exact graph and kernel,
including executed work where vertex IDs can affect convergence or
compression.

## Running example

The [GraphBrew Running Example](GraphBrew-Running-Example) follows one graph
through:

1. graph and CSR input;
2. community membership;
3. contiguous block placement;
4. local vertex ordering;
5. permutation validation and CSR relocation; and
6. resulting property-access locality.

The example explains semantics and does not imply runtime composition search.

## Measurement contract

Report separately:

```text
mapping generation
permutation validation
CSR relocation
kernel-only time
mapping + reuse x kernel
```

Always record the input-label baseline, exact ordered `-o` specification,
mapping fingerprint, graph and binary provenance, scheduler state, and
semantic verification result.
