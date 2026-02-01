# GraphBrew Documentation Index

## Top-Level Guides
- `README.md` — Quick start, CLI overview
- `wiki/` — Detailed guides (Quick Start, Command-Line Reference, Benchmarks)

## Include Structure
```
bench/include/
├── graphbrew/              # GraphBrew extensions
│   ├── graphbrew.h         # Umbrella header (includes everything)
│   ├── reorder/            # Reordering algorithms
│   └── partition/          # Partitioning (trust.h, cagra/popt.h)
├── external/               # External libraries (bundled)
│   ├── gapbs/              # Core GAPBS runtime (builder, graph, etc.)
│   ├── rabbit/             # RabbitOrder
│   ├── gorder/             # GOrder
│   ├── corder/             # COrder
│   └── leiden/             # GVE-Leiden
└── cache_sim/              # Cache simulation
```

## Core C++ Modules
- `bench/include/external/gapbs/` — Core GAPBS runtime (builder.h, graph.h, benchmark.h, etc.)
- `bench/include/graphbrew/` — GraphBrew extensions (graphbrew.h umbrella, reorder/, partition/)
- `bench/include/graphbrew/reorder/` — Reordering algorithms (basic, hub, rabbit, leiden, adaptive, graphbrew)
- `bench/include/graphbrew/partition/` — TRUST partitioning (`trust.h`), Cagra/P-OPT (`cagra/popt.h`)
- `bench/include/cache_sim/` — **Cache simulation** (L1/L2/L3 policies; `cache_sim.h`, `graph_sim.h`)

## External Modules
- `bench/include/external/gapbs/` — Core GAPBS (builder, graph, benchmark, command_line, etc.)
- `bench/include/external/rabbit/` — RabbitOrder community clustering
- `bench/include/external/gorder/` — GOrder graph ordering
- `bench/include/external/corder/` — COrder cache-aware ordering
- `bench/include/external/leiden/` — GVE-Leiden community detection

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

## Conventions
- CLI `-j type:n:m`
  - `0` = Cagra/GraphIT (`MakeCagraPartitionedGraph`, honors `-z`) 
  - `1` = TRUST (`TrustPartitioner::MakeTrustPartitionedGraph`)
- CLI `-o` reordering IDs (0–17) — see `wiki/Command-Line-Reference.md`
