# GraphBrew documentation index

## Public story

- [`README.md`](../README.md) — purpose, architecture, validated result, build,
  and experiment entry points
- [`wiki/Home.md`](../wiki/Home.md) — documentation navigation
- [`wiki/GraphBrewOrder.md`](../wiki/GraphBrewOrder.md) — explicit composition
  axes and configuration
- [`wiki/AdaptiveOrder.md`](../wiki/AdaptiveOrder.md) — deterministic runtime
  policy and legacy boundary
- [`wiki/All-Kernel-Low-Reuse-Selector.md`](../wiki/All-Kernel-Low-Reuse-Selector.md)
  — mechanism and frozen validation
- [`wiki/Reordering-Figure-Catalog.md`](../wiki/Reordering-Figure-Catalog.md)
  — one transformation figure per algorithm ID

## Evidence

- [`recommendation-evidence.json`](recommendation-evidence.json) — aggregate
  evidence and source hashes for recommendation claims
- [`allkernel-lowreuse-evidence.json`](allkernel-lowreuse-evidence.json) —
  30-graph derivation and holdout rows for the frozen policy
- [`figures/graphbrew-architecture.svg`](figures/graphbrew-architecture.svg) —
  canonical architecture figure
- [`figures/graphbrew-lowreuse-policy.svg`](figures/graphbrew-lowreuse-policy.svg)
  — history-free selector decision and Rabbit fallback semantics
- [`figures/graphbrew-leiden-transform.svg`](figures/graphbrew-leiden-transform.svg)
  — graph to community-membership transformation
- [`figures/graphbrew-sizedesc-transform.svg`](figures/graphbrew-sizedesc-transform.svg)
  — community membership to contiguous block ranges
- [`figures/graphbrew-gorder-transform.svg`](figures/graphbrew-gorder-transform.svg)
  — small-community Gorder8 transformation
- [`figures/graphbrew-bfs-transform.svg`](figures/graphbrew-bfs-transform.svg)
  — large-community BFS transformation
- [`figures/graphbrew-cd-parallel.svg`](figures/graphbrew-cd-parallel.svg) —
  serial versus parallel community detection
- [`figures/graphbrew-sgmb4096.svg`](figures/graphbrew-sgmb4096.svg) —
  batched internal super-node moves
- [`figures/graphbrew-gordf5000.svg`](figures/graphbrew-gordf5000.svg) —
  community-size Gorder/BFS decision
- [`figures/graphbrew-norefine.svg`](figures/graphbrew-norefine.svg) —
  refinement bypass
- [`figures/reordering/manifest.json`](figures/reordering/manifest.json) —
  algorithm-to-SVG/editable-source index
- [`figures/editable/README.md`](figures/editable/README.md) — Lucidchart
  import and manual-editing workflow

Detailed raw matrices and large mappings are external artifacts; public claims
are checked against the manifests above.

## Repository map

```text
bench/src/                         canonical graph kernels
bench/src_sim/                     cache-instrumented kernels
bench/include/graphbrew/reorder/   ordering implementations and policies
bench/include/graphbrew/partition/ partitioning implementations
bench/include/external/gapbs/      graph builder and benchmark lifecycle
bench/include/external/            bundled comparison implementations
bench/include/cache_sim/           cache simulator
scripts/graphbrew_experiment.py    public experiment orchestrator
scripts/experiments/               frozen and restartable campaigns
scripts/lib/                       shared experiment infrastructure
scripts/test/                      regression and evidence checks
wiki/                              detailed documentation source
```

## Key implementation files

| File | Role |
|---|---|
| `bench/include/graphbrew/reorder/reorder.h` | algorithm dispatcher and variant resolution |
| `bench/include/graphbrew/reorder/reorder_graphbrew.h` | GraphBrew composition and GVE-Leiden mechanisms |
| `bench/include/graphbrew/reorder/reorder_graphbrew_diagnostics.h` | Callable diagnostic GraphBrew ordering families |
| `bench/include/graphbrew/reorder/reorder_graphbrew_parser.h` | GraphBrew option parser |
| `bench/include/graphbrew/reorder/reorder_adaptive.h` | deterministic rules and retained offline-model modes |
| `bench/include/graphbrew/reorder/reorder_rabbit.h` | CSR and Boost Rabbit |
| `bench/include/graphbrew/reorder/reorder_gorder.h` | faithful and relaxed Gorder variants |
| `scripts/experiments/vldb/` | publication campaign SSOT |

## Interfaces

- `-o 12:<configuration>` is an explicit hand-configured composition.
- `-o 14:_:_:_:allkernel-lowreuse-rule:best-endtoend:<reuse>` is the
  validated deterministic reuse-1/2 policy.
- Historical perceptron, decision-tree, and emulator code remains available
  for research compatibility but is not the validated deployed contribution.

## Validation

```bash
make check
```

`make check` is the authoritative core build, native-test, include-lint, and
Python regression gate.
