# Evidence and Claims

GraphBrew treats a persistent vertex permutation as an executable layout
expression:

```text
<partitioner P, block order B, intra-block order L>

pi(v) = block_offset(B(P(v))) + L[P(v)](v)
```

This is the paper’s novelty boundary: GraphBrew composes the vertex-ID layout
itself, then separates **ordering quality**, **mapping cost**, **executed
work**, and **amortized end-to-end time**. It does not claim the first
community-plus-local ordering or a successful automatic selector.

[![GraphBrew evidence boundary](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-evidence-boundary.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-evidence-boundary.svg?v=graphbrew-public-v4)

The detailed evidence follows the contribution order:

1. [What composition proves](#what-composition-proves)
2. [Why selected compositions work](#why-selected-compositions-work)
3. [Primary practical performance claim](#primary-performance-claim-gorder-quality-at-lower-cost)
4. [Construction optimization](#confirmed-construction-optimization)

## Relation to closest systems

| Prior work | What it composes | GraphBrew distinction |
|---|---|---|
| [GraphIt](https://arxiv.org/abs/1805.00923) | Execution schedules: traversal direction, parallelism, blocking, NUMA, cache, and data layout | Composes the persistent vertex-ID permutation and accounts for its construction and reuse |
| Rabbit Order / Corder / [ReBO](https://doi.org/10.1109/IPDPS65963.2026.00088) | Fixed or tightly integrated multistage pipelines | Exposes partition, block order, and intra-block order as independently addressable operators |
| [Leiden+LLP](https://arxiv.org/abs/2605.21510) | One fixed community-plus-local ordering for graph compression | Provides a general graph-analytics layout expression and fixed-membership cross-kernel mechanism tests |

GraphBrew therefore does not claim the first multistage ordering. Its claim is
the explicit three-stage vertex-layout model and the causal isolation of its
kernel-specific effects.

## Primary performance claim: GORDER-quality at lower cost

**Name:** LeidenGVE–SizeDesc–LocalGorder8

```text
12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8
```

Fresh semantics-v4 confirmation covers 11 graphs, seven kernels, 77
graph-kernel cells, 88 mapping executions, and 385 passing verification cells.
`LocalGorder8` names GraphBrew’s relaxed per-community heuristic; the
standalone faithful comparator is `GORDER_csr`.

| Metric | Result |
|---|---:|
| GORDER_csr / GraphBrew kernel GM | 1.052x |
| GraphBrew / GORDER_csr mapping GM | 0.752x |
| GORDER_csr / GraphBrew end-to-end GM at reuse 1 | 1.332x |
| GraphBrew / GORDER_csr summed kernel seconds | 0.874x |

This is the defensible practical result: GraphBrew improves both kernel
quality and construction cost relative to faithful standalone GORDER_csr.

## Rabbit is the Pareto limitation

| Metric | CSR | Boost |
|---|---:|---:|
| Rabbit / GraphBrew per-cell kernel GM | 1.042x | 1.044x |
| GraphBrew / Rabbit mapping GM | 18.52x | 17.35x |
| Rabbit / GraphBrew GM without Afforest CC | 0.982x | 0.983x |
| GraphBrew / Rabbit summed kernel seconds | 1.243x | 1.242x |

The small Rabbit geometric-mean margin is driven by order-dependent Afforest
CC work. It does not justify the preprocessing overhead, and summed
end-to-end seconds never cross. The paper therefore treats Rabbit as the
practical low-overhead Pareto point, not as a GraphBrew performance victory.

## Confirmed construction optimization

**Name:** Compact-and-Emit

```text
12:leiden:compose:sg_none:comm_identity:
intra_bfs_compact_direct:cd_parallel:sgmb4096:norefine:1:1
```

[![Compact-and-Emit](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-compact-emit.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-compact-emit.svg?v=graphbrew-public-v4)

Compact-and-Emit:

- compacts active one-pass community IDs;
- schedules active communities rather than the largest label range;
- writes final IDs during intra-community BFS; and
- preserves the conventional BFS permutation.

Its five-graph mapping GM is 0.479x the faster Rabbit implementation. The
mandatory ORIGINAL audit found no low-reuse region that beats both doing
nothing and Rabbit. The claim is therefore faster **construction**, not better
ordering quality or a balanced end-to-end arm.

## What composition proves

The corrected historical atlas first exposed the headroom. A subsequent
five-trial confirmation used 10 sealed graphs, five kernels, seven
Rabbit-free compositions, and 550 passing semantic-verification cells.

| Observation | Result |
|---|---:|
| Distinct compositions winning at least one sealed cell | 7 of 7 |
| Distinct winners within each sealed graph | 3–5 |
| Distinct winners within each kernel | 2–5 |
| Cell oracle / best fixed GraphBrew arm | 1.229x |
| Fastest comparator / cell oracle | 1.116x |
| Fastest comparator / post-selected type+kernel table | 1.111x |

These results establish that composition is useful and that one uniform
GraphBrew recipe leaves measurable performance on the table. The best
composition changes even within the same graph when the kernel changes.

They do not establish automatic selection. The family+kernel rule was frozen
before the rapid holdout and then evaluated without changes on the sealed
cohort. It reaches only 0.896x versus the fastest comparator. Against its
predeclared uniform GraphBrew control, the point estimate is 1.035x but the
graph-block interval is [0.984, 1.082] and the worst graph is 0.879x. It
improves GORDER_csr kernels by 1.175x [1.053, 1.330], but its higher mapping
cost delays end-to-end crossover to reuse 67. The result is therefore an
expressiveness certificate, not a deployable graph-type selector.

## Why selected compositions work

A separate fixed-membership factorial keeps the Leiden partition identical
and crosses two block orders with three intra-community layouts. This removes
partition quality as a confounder.

| Contrast | Timing result | Mechanism result |
|---|---:|---|
| BFS: LocalGorder8 vs HubSort | 1.143x [1.039, 1.262] | LocalGorder8 examines slightly more edges but is 1.159x faster per edge; cache/timing direction agrees on 8/10 graphs with rank correlation 0.842 |
| BFS: LocalGorder8 vs RCMpp | 1.133x [1.021, 1.336] | LocalGorder8 examines 0.893x as many edges; per-edge time is unresolved |
| CC: LocalGorder8 vs HubSort | 1.248x [1.074, 1.448] | HubSort performs 1.436x as many compression steps |
| CC: LocalGorder8 vs RCMpp | 1.514x [1.300, 1.772] | LocalGorder8 performs 0.405x as many compression steps, overcoming slower time per step |

Block order has no universal main effect: SizeDesc over DegreeDesc is 1.017x
with interval [0.972, 1.067]. PR favors HubSort by 1.332x over LocalGorder8,
but the trace model reports more L1 misses for HubSort and nearly identical
L3 traffic. Sampled edge span and a post-hoc dynamic-chunk balance proxy also
fail. PR therefore remains a real timing result with an unresolved hardware
mechanism; CC-SV is likewise not assigned a mechanism.

The earlier reuse-1/2 Rabbit-fallback rule remains reproducible through
Algorithm 14, but it is a competitor-backed diagnostic and its public
portfolio accounting did not include selector feature time. It is not a paper
contribution.

## Invalidated evidence

A terminal selector timing campaign inherited `SCHED_IDLE` and nice 19.
Every performance row from that campaign is invalidated. Its 960 semantic/work
verification cells remain verification evidence only.

The runner now records scheduler, nice value, and affinity, and rejects final
timing unless the process is `SCHED_OTHER` at nice 0.

## Machine-readable source

[`docs/recommendation-evidence.json`](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/recommendation-evidence.json)
contains exact values, artifact paths, and SHA-256 hashes.
