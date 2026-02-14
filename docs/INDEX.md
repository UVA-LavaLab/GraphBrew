# GraphBrew Documentation Index

## Top-Level Guides
- `README.md` — Quick start, CLI overview
- `wiki/` — Detailed guides (Quick Start, Command-Line Reference, Benchmarks)

## Include Structure
```
bench/include/
├── graphbrew/                  # GraphBrew extensions
│   ├── graphbrew.h             # Umbrella header (includes everything)
│   ├── reorder/                # Reordering algorithms
│   │   ├── reorder.h           # Main dispatcher (resolveVariant, hasVariants, etc.)
│   │   ├── reorder_types.h     # Enums, perceptron weights, variant resolution
│   │   ├── reorder_basic.h     # ORIGINAL, Sort, Random
│   │   ├── reorder_hub.h       # HubSort, HubCluster, DBG, HubSortDBG, HubClusterDBG
│   │   ├── reorder_classic.h   # COrder (10)
│   │   ├── reorder_rabbit.h    # RabbitOrder CSR (8:csr) + Boost (8:boost)
│   │   ├── reorder_gorder.h    # GOrder CSR (9:csr) + parallel (9:fast)
│   │   ├── reorder_rcm.h       # RCM default + BNF (11:bnf)
│   │   ├── reorder_graphbrew.h # GraphBrewOrder (12) — Leiden + per-community pipeline
│   │   └── reorder_adaptive.h  # AdaptiveOrder (14) — perceptron-based selection
│   └── partition/              # Partitioning
│       ├── trust.h             # TRUST partitioning
│       └── cagra/popt.h        # Cagra/P-OPT partitioning
├── external/                   # External libraries (bundled)
│   ├── gapbs/                  # Core GAPBS runtime (builder, graph, benchmark, cli)
│   ├── rabbit/                 # RabbitOrder community clustering
│   ├── gorder/                 # GOrder graph ordering (GoGraph baseline)
│   ├── corder/                 # COrder cache-aware ordering
│   └── leiden/                 # GVE-Leiden community detection
└── cache_sim/                  # Cache simulation (cache_sim.h, graph_sim.h)
```

## Core C++ Modules
- `bench/include/external/gapbs/` — GAPBS runtime (builder.h, graph.h, benchmark.h, command_line.h, etc.)
- `bench/include/graphbrew/` — GraphBrew extensions (graphbrew.h umbrella, reorder/, partition/)
- `bench/include/graphbrew/reorder/` — All reordering algorithms (0–15), variant dispatch, perceptron weights
- `bench/include/graphbrew/partition/` — TRUST partitioning (`trust.h`), Cagra/P-OPT (`cagra/popt.h`)
- `bench/include/cache_sim/` — Cache simulation (L1/L2/L3 policies; `cache_sim.h`, `graph_sim.h`)

## External Libraries
- `bench/include/external/rabbit/` — RabbitOrder community clustering
- `bench/include/external/gorder/` — GOrder graph ordering (GoGraph reference)
- `bench/include/external/corder/` — COrder cache-aware ordering
- `bench/include/external/leiden/` — GVE-Leiden community detection

## Python Tooling
- `scripts/graphbrew_experiment.py` — Main orchestration pipeline (reorder, benchmark, cache, weights)
- `scripts/adaptive_emulator.py` — AdaptiveOrder emulator and evaluation
- `scripts/perceptron_experiment.py` — Perceptron training experiments
- `scripts/quick_cache_compare.py` — Quick cache miss comparison
- `scripts/lib/` — Shared modules (utils, reorder, features, weights, training, analysis, etc.)
- `scripts/test/` — Pytest suite (algorithm variants, cache sim, weights, GraphBrew experiment)

## Tooling
- `make lint-includes` — check for legacy includes
- `python3 scripts/check_includes.py` — same as above

## Conventions
- CLI `-j type:n:m`
  - `0` = Cagra/GraphIT (`MakeCagraPartitionedGraph`, honors `-z`) 
  - `1` = TRUST (`TrustPartitioner::MakeTrustPartitionedGraph`)
- CLI `-o` reordering IDs (0–15) — see `wiki/Command-Line-Reference.md`
- Variants via colon: `-o 9:fast`, `-o 8:boost`, `-o 11:bnf`, `-o 12:leiden`
