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

## Evidence

- [`recommendation-evidence.json`](recommendation-evidence.json) — aggregate
  evidence and source hashes for recommendation claims
- [`allkernel-lowreuse-evidence.json`](allkernel-lowreuse-evidence.json) —
  30-graph derivation and holdout rows for the frozen policy
- [`figures/graphbrew-architecture.svg`](figures/graphbrew-architecture.svg) —
  canonical architecture figure
- [`figures/graphbrew-lowreuse-policy.svg`](figures/graphbrew-lowreuse-policy.svg)
  — history-free selector decision and Rabbit fallback semantics
- [`figures/graphbrew-relabeling-example.svg`](figures/graphbrew-relabeling-example.svg)
  — topology-preserving relabeling example
- [`figures/graphbrew-cost-controls.svg`](figures/graphbrew-cost-controls.svg) —
  `cd_parallel`, `sgmb4096`, `gordf5000`, and `norefine`

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
