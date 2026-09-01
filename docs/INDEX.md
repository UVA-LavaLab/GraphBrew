# GraphBrew documentation index

## Start here

- [`README.md`](../README.md) — project overview, build, and experiment entry
  point
- [`wiki/GraphBrew-Running-Example.md`](../wiki/GraphBrew-Running-Example.md) —
  one graph through partition, layout, relabeling, and locality
- [`wiki/GraphBrewOrder.md`](../wiki/GraphBrewOrder.md) — explicit composition
  grammar and stage semantics
- [`wiki/Reproducible-Experiments.md`](../wiki/Reproducible-Experiments.md) —
  generic measurement workflow

## Public figures

- [`figures/graphbrew-architecture.svg`](figures/graphbrew-architecture.svg) —
  explicit layout expression and six-stage pipeline
- [`figures/graphbrew-compact-emit.svg`](figures/graphbrew-compact-emit.svg) —
  sparse-ID compaction and direct emission
- [`figures/reordering/manifest.json`](figures/reordering/manifest.json) —
  measured output order and editable source for every algorithm ID

Large raw matrices and mappings stay outside the repository.

## Repository map

```text
bench/src/                         graph kernels
bench/src_sim/                     cache-instrumented kernels
bench/include/graphbrew/reorder/   ordering implementations
bench/include/graphbrew/partition/ partitioning and shard support
bench/include/external/gapbs/      graph builder and benchmark lifecycle
scripts/graphbrew_experiment.py    public experiment orchestrator
scripts/experiments/               specialized restartable campaigns
scripts/lib/                       shared experiment infrastructure
scripts/test/                      regression checks
wiki/                              documentation source
```

## Interfaces

- `-o 12:<configuration>` runs one explicit GraphBrew composition.
- `-o 13:<mapping>` loads a pre-generated permutation.
- `-o 14` remains an experimental compatibility interface.

## Validation

```bash
python3 scripts/generate_public_figures.py --check
make check
```
