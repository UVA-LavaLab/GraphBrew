# GraphBrew Documentation Index

## Project Folder Hierarchy
```
GraphBrew/
├── bench/                          # C++ benchmark suite
│   ├── src/                        # Algorithm source (pr, bfs, sssp, cc, cc_sv, pr_spmv, bc, tc)
│   ├── src_sim/                    # Cache-instrumented versions (8 algorithms)
│   ├── bin/                        # Compiled benchmark binaries
│   ├── bin_sim/                    # Compiled simulation binaries
│   └── include/                    # Headers (see Include Structure below)
├── scripts/                        # Python experiment infrastructure
│   ├── graphbrew_experiment.py      # Main pipeline entry point (single top-level script)
│   ├── experiments/                 # Paper experiment suites (VLDB + ECG)
│   ├── lib/                         # 5 sub-packages (core, pipeline, ml, analysis, tools)
│   └── test/                        # pytest test suite
├── wiki/                           # 24 documentation pages
├── docs/                           # Quick guides + INDEX.md
├── research/                       # Paper drafts and reference materials
├── Makefile                        # Build system (make all, make all-sim)
├── build_wsl.ps1                   # WSL build helper
└── setup_wsl.ps1                   # WSL setup (dependencies + Boost 1.58)
```

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
└── cache_sim/                  # Cache simulation
    ├── cache_sim.h             # 9 eviction policies (LRU,FIFO,RANDOM,LFU,PLRU,SRRIP,GRASP,P-OPT,ECG)
    ├── graph_sim.h             # Graph wrappers + SIM_CACHE_READ/WRITE/SET_VERTEX macros
    └── graph_cache_context.h   # Unified context: PropertyRegion, FatIDConfig, GraphTopology
```

## Core C++ Modules
- `bench/include/external/gapbs/` — GAPBS runtime (builder.h, graph.h, benchmark.h, command_line.h, etc.)
- `bench/include/graphbrew/` — GraphBrew extensions (graphbrew.h umbrella, reorder/, partition/)
- `bench/include/graphbrew/reorder/` — All reordering algorithms (0–15), variant dispatch, perceptron weights
- `bench/include/graphbrew/partition/` — TRUST partitioning (`trust.h`), Cagra/P-OPT (`cagra/popt.h`)
- `bench/include/cache_sim/` — Cache simulation: 9 eviction policies, `GraphCacheContext` (multi-region 4-tier classification, `FatIDConfig` adaptive fat-ID encoding), `graph_sim.h` macros

## External Libraries
- `bench/include/external/rabbit/` — RabbitOrder community clustering
- `bench/include/external/gorder/` — GOrder graph ordering (GoGraph reference)
- `bench/include/external/corder/` — COrder cache-aware ordering
- `bench/include/external/leiden/` — GVE-Leiden community detection

## Python Tooling
- `scripts/graphbrew_experiment.py` — Main orchestration pipeline (reorder, benchmark, cache)
- `scripts/lib/` — 5 sub-packages (core, pipeline, ml, analysis, tools); see `scripts/lib/README.md`
- `scripts/lib/ml/adaptive_emulator.py` — AdaptiveOrder emulator and evaluation
- `scripts/lib/core/datastore.py` — Unified data store (BenchmarkStore, GraphPropsStore)
- `scripts/test/` — Pytest suite (algorithm variants, cache sim, weights, GraphBrew experiment)

## Tooling
- `make lint-includes` — check for legacy includes
- `python3 -m scripts.lib.tools.check_includes` — same as above

## Conventions
- CLI `-j type:n:m`
  - `0` = Cagra/GraphIT (`MakeCagraPartitionedGraph`, honors `-z`) 
  - `1` = TRUST (`TrustPartitioner::MakeTrustPartitionedGraph`)
- CLI `-o` reordering IDs (0–15) — see `wiki/Command-Line-Reference.md`
- Variants via colon: `-o 9:fast`, `-o 8:boost`, `-o 11:bnf`, `-o 12:leiden`
