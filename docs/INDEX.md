# GraphBrew Documentation Index

## Top-Level Guides
- `README.md` — Quick start, CLI overview
- `wiki/` — Detailed guides (Quick Start, Command-Line Reference, Benchmarks)

## Core C++ Modules
- `bench/include/graphbrew/` — Umbrella headers (`graphbrew.h`, builder, reorder, partition, util)
- `bench/include/graphbrew/builder.h` — Graph construction, reordering, partitioning
- `bench/include/graphbrew/reorder/` — Reordering algorithms (basic, hub, rabbit, leiden, adaptive, graphbrew)
- `bench/include/graphbrew/partition/` — TRUST partitioning (`trust.h`), Cagra/P-OPT (`cagra/popt.h` → `graphSlicer`, `MakeCagraPartitionedGraph`)
- `bench/include/cache_sim/` — **Cache simulation** (L1/L2/L3 policies, instrumentation; `cache_sim.h`, `graph_sim.h`)

## External Modules
- `bench/include/external/rabbit/`
- `bench/include/external/gorder/`
- `bench/include/external/corder/`
- `bench/include/external/leiden/`

## Python Tooling
- `scripts/graphbrew_experiment.py` — Orchestration pipeline (reorder, benchmark, cache, weights)
- `scripts/lib/` — Python modules (phases, features, weights, analysis)
- `scripts/test/` — Pytest suite with tiny graphs

## Partitioning & Cache
- **Cagra/GraphIT partitioning**: `bench/include/graphbrew/partition/cagra/popt.h`
- **TRUST partitioning**: `bench/include/graphbrew/partition/trust.h`
- **Cache simulation**: `bench/include/cache_sim/`

## Tooling
- `make lint-includes` — check for legacy includes
- `python3 scripts/check_includes.py` — same as above

## Symbols Map
See `docs/SYMBOLS.md` for function → file mappings.

## Conventions
- CLI `-j type:n:m`
  - `0` = Cagra/GraphIT (`MakeCagraPartitionedGraph`, honors `-z`) 
  - `1` = TRUST (`TrustPartitioner::MakeTrustPartitionedGraph`)
- CLI `-o` reordering IDs (0–17) — see `wiki/Command-Line-Reference.md`
