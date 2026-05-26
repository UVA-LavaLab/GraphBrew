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
├── wiki/                           # Detailed documentation pages
├── docs/                           # Quick guides + INDEX.md
├── research/                       # Paper drafts and reference materials
├── Makefile                        # Build system (make all, make all-sim)
├── build_wsl.ps1                   # WSL build helper
└── setup_wsl.ps1                   # WSL setup (dependencies + Boost 1.58)
```

## Top-Level Guides
- `README.md` — Quick start, CLI overview
- `wiki/` — Detailed guides (Quick Start, Command-Line Reference, Benchmarks)
- `wiki/ECG-Final-Runs.md` — current ECG/gem5 final-run profiles, charged P-OPT, graph checks, and status commands

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
- `bench/include/cache_sim/` — Cache simulation: core eviction policies plus graph-aware context; final ECG runners add aliases such as `POPT_CHARGED` by changing effective cache geometry and output metadata

## External Libraries
- `bench/include/external/rabbit/` — RabbitOrder community clustering
- `bench/include/external/gorder/` — GOrder graph ordering (GoGraph reference)
- `bench/include/external/corder/` — COrder cache-aware ordering
- `bench/include/external/leiden/` — GVE-Leiden community detection

## Python Tooling
- `scripts/graphbrew_experiment.py` — Main orchestration pipeline (reorder, benchmark, cache)
- `scripts/experiments/ecg/paper_pipeline.py` — ECG paper workflow wrapper (profile runs, aggregate CSVs, SVG figures, LaTeX tables)
- `scripts/experiments/ecg/final_paper_run.py` — ECG final-run harness (manifest, graph checks, validation gate, resume/status, combined CSVs)
- `scripts/experiments/ecg/roi_matrix.py` — cache_sim/gem5/Sniper ROI policy matrix runner, including charged P-OPT labels and ECG_PFX prefetcher rows
- `scripts/experiments/ecg/proof_matrix.py` — cache_sim ECG component ablation runner
- `bench/src_sim/ecg_preprocess.cc` — no-simulation ECG preprocessing overhead benchmark for degree scan, P-OPT matrix construction, and ECG mask/PFX construction
- `wiki/ECG-Slurm-Runs.md` — UVA Slurm workflow for split graph/benchmark/policy shards and post-hoc aggregation
- `wiki/ECG-Sniper-Runs.md` — Sniper setup, overlay, smoke, thread-surface, DROPLET, and ECG_PFX status workflow
- `plans/sniper-sim-integration-plan.md` — Sniper backend plan for scalable multicore ECG validation
- `scripts/setup_sniper.py` — Sniper checkout/build scaffold; upstream clone lives under `bench/include/sniper_sim/snipersim/`
- `scripts/setup_gem5.py` — gem5 checkout/build scaffold; X86 and RISCV overlays include ECG_PFX and `ecg.extract`
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
