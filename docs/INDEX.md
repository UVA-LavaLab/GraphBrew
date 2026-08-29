# GraphBrew documentation index

## Start here

- [`README.md`](../README.md) — project story, confirmed claims, build, and
  experiment entry point
- [`wiki/Evidence-and-Claims.md`](../wiki/Evidence-and-Claims.md) — exact
  evidence boundary and rejected claims
- [`wiki/GraphBrew-Running-Example.md`](../wiki/GraphBrew-Running-Example.md) —
  one graph through partition, layout, relabeling, and locality
- [`wiki/GraphBrewOrder.md`](../wiki/GraphBrewOrder.md) — explicit composition
  grammar and the two paper contributions
- [`wiki/Reproducible-Experiments.md`](../wiki/Reproducible-Experiments.md) —
  frozen measurement workflow

## Public evidence

- [`recommendation-evidence.json`](recommendation-evidence.json) — claim
  values, source artifact hashes, and rejected claims
- [`figures/graphbrew-architecture.svg`](figures/graphbrew-architecture.svg) —
  explicit pipeline, quality arm, and construction optimization
- [`figures/graphbrew-evidence-boundary.svg`](figures/graphbrew-evidence-boundary.svg)
  — confirmed and rejected claims
- [`figures/graphbrew-compact-emit.svg`](figures/graphbrew-compact-emit.svg) —
  sparse-ID compaction and direct emission
- [`figures/reordering/manifest.json`](figures/reordering/manifest.json) —
  measured output order and editable source for every algorithm ID

Large raw matrices and mappings stay in the external artifact root. Public
claims are checked against `recommendation-evidence.json`.

## Repository map

```text
bench/src/                         graph kernels
bench/src_sim/                     cache-instrumented kernels
bench/include/graphbrew/reorder/   ordering implementations
bench/include/graphbrew/partition/ partitioning and shard support
bench/include/external/gapbs/      graph builder and benchmark lifecycle
scripts/graphbrew_experiment.py    public experiment orchestrator
scripts/experiments/               frozen and restartable campaigns
scripts/lib/                       shared experiment infrastructure
scripts/test/                      regression and evidence checks
wiki/                              documentation source
```

## Interfaces

- `-o 12:<configuration>` runs one explicit GraphBrew composition.
- `-o 13:<mapping>` loads a pre-generated permutation.
- `-o 14` remains an experimental compatibility interface; it is not a
  headline paper contribution.

## Validation

```bash
python3 scripts/generate_public_figures.py --check
make check
```
