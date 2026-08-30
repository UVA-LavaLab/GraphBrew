# Evidence and Claims

GraphBrew separates **ordering quality**, **mapping cost**, and **amortized
end-to-end time**. A claim is included only when its baseline, workload,
mapping policy, and verification cohort are explicit.

[![GraphBrew evidence boundary](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-evidence-boundary.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-evidence-boundary.svg?v=graphbrew-public-v4)

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
