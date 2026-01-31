# GraphBrew Documentation Index

## Top-Level Guides
- `README.md` — Quick start, CLI overview
- `wiki/` — Detailed guides (Quick Start, Command-Line Reference, Benchmarks)

## Core C++ Modules
- `bench/include/gapbs/builder.h` — Graph construction, reordering, partitioning
- `bench/include/gapbs/reorder/` — Reordering algorithms (basic, hub, rabbit, leiden, adaptive, graphbrew)
- `bench/include/gapbs/partition/` — TRUST partitioning (`trust.h`)
- `bench/include/gapbs/cache/` — **Cagra/P-OPT partition helpers** (`popt.h` → `graphSlicer`, `MakeCagraPartitionedGraph`)
- `bench/include/cache/` — **Cache simulation** (L1/L2/L3 policies, instrumentation; `cache_sim.h`, `graph_sim.h`)

## Python Tooling
- `scripts/graphbrew_experiment.py` — Orchestration pipeline (reorder, benchmark, cache, weights)
- `scripts/lib/` — Python modules (phases, features, weights, analysis)
- `scripts/test/` — Pytest suite with tiny graphs

## Partitioning & Cache
- **Cagra/GraphIT partitioning**: `bench/include/gapbs/cache/popt.h`
- **TRUST partitioning**: `bench/include/gapbs/partition/trust.h`
- **Cache simulation**: `bench/include/cache/`

## Symbols Map
See `docs/SYMBOLS.md` for function → file mappings.

## Conventions
- CLI `-j type:n:m`
  - `0` = Cagra/GraphIT (`MakeCagraPartitionedGraph`, honors `-z`) 
  - `1` = TRUST (`TrustPartitioner::MakeTrustPartitionedGraph`)
- CLI `-o` reordering IDs (0–17) — see `wiki/Command-Line-Reference.md`
