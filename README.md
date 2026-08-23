# GraphBrew ReusePlan Cache Architecture

GraphBrew explores graph-aware cache management for irregular graph analytics.
The design carries compact reuse information with each streamed edge, attaches
that information to the corresponding property request, and uses it to guide
last-level-cache replacement and placement.

The current architecture combines three mechanisms:

- **ReusePlan records** carry a reuse tier and the next two property-reuse epochs.
- **ReuseBind** binds that metadata to the exact property load.
- **FlowThrough** keeps one-touch edge records from occupying the shared LLC
  after a miss while preserving private-cache fills and LLC hits.

## Documentation

- [Illustrated design guide](wiki/ReusePlan-FlowThrough.md) — start here for the
  mechanism, worked examples, and simulator mapping.
- [Evaluation methodology](wiki/Evaluation-Methodology.md) — workloads,
  baselines, metrics, and reporting rules.
- [Build and reproduction guide](wiki/Reproduction.md) — datasets, builds,
  tests, and experiment commands.
- [Repository hygiene](wiki/Repository-Hygiene.md) — what belongs in a push
  and what must stay local.

Performance tables are intentionally omitted until the final evaluation is
complete.

## Repository layout

| Path | Purpose |
|---|---|
| `bench/include/` | Shared cache policy, metadata, and simulator integration |
| `bench/src_sim/` | Functional cache-simulator graph kernels |
| `bench/src_gem5/` | gem5 graph kernels |
| `bench/src_sniper/` | Sniper graph workload |
| `bench/src_rtl/` | Synthesizable ReusePlan physical-cost models and testbenches |
| `scripts/experiments/ecg/` | Experiment runners and analysis tools |
| `scripts/test/` | Unit, integration, and documentation checks |
| `wiki/` | Illustrated design, methodology, and reproduction documentation |

Generated graphs, binaries, traces, and experiment output under `results/` are
not tracked.
