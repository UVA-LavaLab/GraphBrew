# Evidence and Claims

GraphBrew separates **ordering quality**, **mapping cost**, and **amortized
end-to-end time**. A claim is included only when its baseline, workload,
mapping policy, and verification cohort are explicit.

[![GraphBrew evidence boundary](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-evidence-boundary.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-evidence-boundary.svg?v=graphbrew-public-v4)

## Confirmed quality point

**Name:** LeidenGVE–SizeDesc–LocalGorder8

```text
12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8
```

Fresh semantics-v4 confirmation covers 11 graphs, seven kernels, 77
graph-kernel cells, 88 mapping executions, and 385 passing verification cells.
`LocalGorder8` names GraphBrew’s relaxed per-community heuristic; the
standalone faithful comparator is `GORDER_csr`.

| Comparator | Comparator/GraphBrew kernel GM | GraphBrew/comparator mapping GM |
|---|---:|---:|
| Rabbit CSR | 1.042x | 18.52x |
| Rabbit Boost | 1.044x | 17.35x |
| GORDER_csr | 1.052x | 0.752x |

The Rabbit result is driven by order-dependent Afforest CC work. Without CC,
the Rabbit geometric means fall below one. Rabbit cell-GM break-even is about
37,000–40,000 reuses, while summed Rabbit seconds never cross.

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

## Why there is no adaptive headline

The corrected historical atlas contains nine complete Rabbit-free GraphBrew
arms:

| Policy | Fastest comparator / GraphBrew |
|---|---:|
| In-sample per-cell oracle | 1.062x |
| Best fixed arm | 0.946x |
| Leave-one-graph-out kernel policy | 0.907x |

The oracle is post-selected headroom. Fixed and graph-held-out policies lose,
so the current library does not support an automated-generator claim.

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
